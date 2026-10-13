"""Supervisor 그래프 흐름 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_graph_flow.py -q

하위 노드를 가짜 함수로 바꿔 라우팅·재작업·Loop·종료만 확인한다. 가짜 노드도 실제 노드와 같은 키 규칙을 따른다
(제어 키는 쓰지 않고, 검증·평가 결과에 현재 report_version을 남긴다).
"""
import pytest
from langgraph.graph import END

import obs
from agents import supervisor
from config import MAX_REPORT_RETRY
from graph import WORKERS, build_graph
from state import CONTROL_KEYS, RESET_ON_NEXT

ANALYSIS_NODES = set(supervisor.ANALYSIS)


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "LOG_DIR", tmp_path / "decisions")


class Scenario:
    def __init__(self, candidates, decisions, eligible=None, tech_gap=False, verify=None, quality=None):
        self.candidates = candidates
        self.decisions = decisions              # {기업: "투자"|"보류"}
        self.eligible = eligible or {}          # {기업: False} 이면 조건 미충족
        self.tech_gap = tech_gap                # 첫 기술 요약에서 개발 단계가 빔 → 재작업 후 채워짐
        self.verify = list(verify or [])        # 사실 검증 통과 여부를 차례로
        self.quality = list(quality or [])      # 품질 평가 통과 여부를 차례로
        self.calls = []

    def _name(self, state):
        return (state.get("company") or {}).get("name", "")

    def _sources(self, state, node):
        name = self._name(state)
        return [{"company": name, "node": node, "publisher": p, "url": f"https://{p}.com"} for p in ("pubA", "pubB")]

    def nodes(self):
        def explorer(state):
            self.calls.append("explorer")
            cands = state.get("candidates") or self.candidates
            idx = state.get("current_index", 0)
            if idx >= len(cands):
                return {**RESET_ON_NEXT, "candidates": cands, "eligibility_reason": "조건을 충족하는 후보를 찾지 못함"}
            name = cands[idx]
            ok = self.eligible.get(name, True)
            return {**RESET_ON_NEXT, "candidates": cands, "company": {"name": name}, "is_eligible": ok,
                    "eligibility_reason": "" if ok else "상장사"}

        def analysis(node, key, value):
            def fn(state):
                self.calls.append(node)
                v = value(state) if callable(value) else value
                return {key: v, "sources": self._sources(state, node)}
            return fn

        def tech_value(state):
            reworked = "tech_summary" in (state.get("rework_request") or {})
            return {"missing_fields": ["development_stage"] if self.tech_gap and not reworked else []}

        def judge(state):
            self.calls.append("judge")
            name = self._name(state)
            decision = self.decisions[name]
            out = {"scores": {"decision": decision, "total": 78 if decision == "투자" else 60}}
            if decision == "보류":
                out["rejected"] = [{"company": name, "reason": "70점 미만", "total": 60}]
            return out

        def reporter(state):
            self.calls.append("reporter")
            return {"report": f"보고서 v{state.get('report_version')}"}

        def verifier(state):
            self.calls.append("verifier")
            passed = self.verify.pop(0) if self.verify else True
            finalized = not passed and state.get("retry_count", 0) >= MAX_REPORT_RETRY
            return {"verify_result": {"passed": passed, "finalized": finalized, "mismatches": [] if passed else [1],
                                      "report_version": state.get("report_version")}}

        def quality(state):
            self.calls.append("quality_eval")
            passed = self.quality.pop(0) if self.quality else True
            return {"eval_result": {"passed": passed, "report_version": state.get("report_version"),
                                    "items": {"coverage": {"passed": passed, "reasons": []}}, "fix_instructions": ""}}

        return {
            "explorer": explorer,
            "tech_summary": analysis("tech_summary", "tech_summary", tech_value),
            "market": analysis("market", "market_analysis", {"info_insufficient": False}),
            "competitor": analysis("competitor", "competitor_analysis", {"competitors": [{"name": "C사"}]}),
            "team": analysis("team", "team_analysis", {"members": [{"name": "홍길동"}], "concerns": []}),
            "judge": judge,
            "reporter": reporter,
            "verifier": verifier,
            "quality_eval": quality,
        }

    def run(self):
        graph = build_graph(self.nodes())
        return graph.invoke({"trace_id": "t"}, {"recursion_limit": 200})


def steps(calls):
    """병렬로 돈 분석 노드 묶음을 하나로 접어 실행 순서를 읽기 쉽게 만든다."""
    out = []
    for c in calls:
        if c in ANALYSIS_NODES:
            if out and isinstance(out[-1], set):
                out[-1].add(c)
                continue
            out.append({c})
        else:
            out.append(c)
    return out


def decisions():
    return [e["decision"] for e in obs.read_events("t", kind="decision")]


# ---------- 그래프 구조 ----------
def test_every_worker_returns_only_to_supervisor():
    g = build_graph().get_graph()
    for e in g.edges:
        if e.source in WORKERS:
            assert e.target == "supervisor", f"{e.source} → {e.target}"
    assert {e.target for e in g.edges if e.source == "supervisor"} == set(WORKERS) | {"__end__"}


# ---------- 시나리오 ----------
def test_first_candidate_invest():
    s = Scenario(["A사", "B사"], {"A사": "투자"})
    out = s.run()
    assert steps(s.calls) == ["explorer", ANALYSIS_NODES, "judge", "reporter", "verifier", "quality_eval"]
    assert out["step_count"] == 7      # 병렬 4개 뒤에도 supervisor는 한 번만 실행
    assert out["report_version"] == 1
    assert decisions()[-1] == "END"


def test_hold_then_next_candidate_invest():
    s = Scenario(["A사", "B사"], {"A사": "보류", "B사": "투자"})
    out = s.run()
    assert steps(s.calls) == ["explorer", ANALYSIS_NODES, "judge", "explorer", ANALYSIS_NODES, "judge",
                              "reporter", "verifier", "quality_eval"]
    assert out["company"]["name"] == "B사" and out["current_index"] == 1
    assert [r["company"] for r in out["rejected"]] == ["A사"]


def test_ineligible_candidate_is_skipped_without_analysis():
    s = Scenario(["A사", "B사"], {"B사": "투자"}, eligible={"A사": False})
    out = s.run()
    assert steps(s.calls)[:3] == ["explorer", "explorer", ANALYSIS_NODES]
    assert out["company"]["name"] == "B사"


def test_insufficient_evidence_triggers_one_rework():
    s = Scenario(["A사"], {"A사": "투자"}, tech_gap=True)
    out = s.run()
    # 실행 기록으로는 병렬 4개와 바로 뒤 재작업 1개의 경계가 안 보여 supervisor 결정 로그로 확인한다
    routes = [set(d) if isinstance(d, list) else d for d in decisions()]
    assert routes == [{"explorer"}, ANALYSIS_NODES, {"tech_summary"}, {"judge"}, {"reporter"}, {"verifier"},
                      {"quality_eval"}, "END"]
    assert s.calls.count("tech_summary") == 2 and s.calls.count("market") == 1
    assert out["rework_count"] == {"tech_summary": 1}
    assert out["evidence_status"]["tech_summary"]["sufficient"] is True
    assert ["tech_summary"] in decisions()


def test_quality_fail_loops_and_reverifies_rewritten_report():
    s = Scenario(["A사"], {"A사": "투자"}, quality=[False, True])
    out = s.run()
    assert steps(s.calls)[-6:] == ["reporter", "verifier", "quality_eval", "reporter", "verifier", "quality_eval"]
    assert out["report_version"] == 2 and out["quality_loops"] == 1
    assert out["report"] == "보고서 v2"


def test_verify_mismatch_rewrites_then_finalizes():
    s = Scenario(["A사"], {"A사": "투자"}, verify=[False, False])
    out = s.run()
    assert steps(s.calls)[-5:] == ["reporter", "verifier", "reporter", "verifier", "quality_eval"]
    assert out["retry_count"] == 1 and out["verify_result"]["finalized"] is True


def test_all_hold_writes_no_recommendation_report():
    s = Scenario(["A사", "B사"], {"A사": "보류", "B사": "보류"})
    out = s.run()
    assert steps(s.calls)[-4:] == ["judge", "reporter", "verifier", "quality_eval"]
    assert len(out["rejected"]) == 2
    reasons = [e["reason"] for e in obs.read_events("t", kind="decision")]
    assert any("후보 소진" in r and "추천 기업 없음" in r for r in reasons)


def test_step_limit_forces_report_and_ends(monkeypatch):
    monkeypatch.setattr(supervisor, "MAX_STEPS", 3)
    s = Scenario(["A사"], {"A사": "투자"})
    out = s.run()
    assert s.calls[-1] == "reporter"           # 사실 검증·품질 평가 없이 강제 마무리
    assert out["last_error"] and out["report"]
    assert out["step_count"] == 5


def test_fake_workers_never_write_control_keys():
    # 가짜 노드가 실제 노드 규칙을 지키는지 (5단계에서 실제 노드에도 같은 검사를 붙인다)
    s = Scenario(["A사", "B사"], {"A사": "보류", "B사": "투자"}, tech_gap=True, quality=[False, True])
    for name, fn in s.nodes().items():
        state = {"candidates": ["A사"], "company": {"name": "A사"}, "report_version": 1, "retry_count": 0}
        if name == "judge":
            continue
        assert not set(fn(state)) & CONTROL_KEYS, name
