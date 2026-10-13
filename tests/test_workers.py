"""하위 노드 규칙 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_workers.py -q

- 실제 하위 노드는 제어 키(state.CONTROL_KEYS)를 반환하지 않는다
- 하위 노드 예외는 guarded()가 노드별 대체 결과와 agent_errors로 바꾸고, 그래프는 끝까지 간다
"""
from types import SimpleNamespace

import pytest
from langgraph.graph import END

import obs
from agents import explorer, judge, reporter, supervisor, team, verifier
from graph import WORKERS, build_graph, guarded
from state import CONTROL_KEYS
from test_graph_flow import Scenario


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "LOG_DIR", tmp_path / "decisions")
    monkeypatch.setattr(judge, "JUDGE_LOG_PATH", tmp_path / "judge_log.jsonl")


class FakeLLM:
    def invoke(self, messages):
        return SimpleNamespace(content="## SUMMARY\n추천 기업 없음.\n")


HOLD_STATE = {
    "candidates": ["A사"], "company": {"name": "A사"}, "sources": [], "report_version": 1,
    "scores": {"decision": "보류", "total": 60, "hold_reasons": ["70점 미만"], "items": {}},
    "rejected": [{"company": "A사", "reason": "70점 미만", "total": 60}],
}


# ---------- 실제 노드는 제어 키를 쓰지 않는다 ----------
def test_explorer_reads_current_index_without_writing_it(monkeypatch):
    monkeypatch.setattr(explorer, "discover_candidates", lambda: ["A사", "B사"])
    monkeypatch.setattr(explorer, "verify_company", lambda name: {
        "company": {"name": name}, "is_eligible": True, "eligibility_reason": "", "sources": []})
    first = explorer.explorer_node({"current_index": 0})
    assert first["company"]["name"] == "A사" and first["candidates"] == ["A사", "B사"]
    second = explorer.explorer_node({"current_index": 1, "candidates": ["A사", "B사"]})
    assert second["company"]["name"] == "B사"     # supervisor가 정한 순번 그대로 (+1 하지 않음)
    assert not (set(first) | set(second)) & CONTROL_KEYS


def test_explorer_with_no_candidates_has_no_company(monkeypatch):
    monkeypatch.setattr(explorer, "discover_candidates", lambda: [])
    out = explorer.explorer_node({"current_index": 0})
    assert out["company"] is None and out["candidates"] == []
    assert not set(out) & CONTROL_KEYS


def test_reporter_does_not_count_or_limit_retries():
    # 한도 관리는 supervisor 몫이라 retry_count가 커도 예외 없이 쓰고, retry_count를 반환하지 않는다
    state = {**HOLD_STATE, "retry_count": 5, "verify_result": {"passed": False, "mismatches": []}}
    out = reporter.reporter_node(state, llm=FakeLLM())
    assert set(out) == {"report"}


def test_verifier_stamps_report_version():
    state = {**HOLD_STATE, "report_version": 3}
    state["report"] = reporter.reporter_node(state, llm=FakeLLM())["report"]
    out = verifier.verifier_node(state, judge_llm=False, retriever=False)
    assert out["verify_result"]["report_version"] == 3
    assert not set(out) & CONTROL_KEYS


def test_team_node_returns_only_payload(monkeypatch):
    monkeypatch.setattr(team, "search_many", lambda *a, **k: [])
    out = team.team_node({"company": {"name": "A사", "ceo": ""}})
    assert set(out) == {"team_analysis", "sources"}


def test_judge_treats_failed_analysis_as_missing():
    assert judge.is_placeholder({"failed": True, "error": "Timeout"})


# ---------- 예외 래퍼 ----------
def boom(state):
    raise TimeoutError("검색 API 응답 없음")


@pytest.mark.parametrize("name", [n for n in WORKERS if n != "reporter"])
def test_guarded_turns_exception_into_fallback(name):
    state = {"trace_id": "t", "candidates": ["A사", "B사"], "current_index": 1, "company": {"name": "B사"},
             "report_version": 2}
    out = guarded(name, boom)(state)
    assert not set(out) & CONTROL_KEYS
    assert out["agent_errors"] == [{"company": "B사", "node": name, "error": "TimeoutError: 검색 API 응답 없음"}]
    assert obs.read_events("t", kind="error")[0]["node"] == name


def test_guarded_fallback_values():
    state = {"candidates": ["A사", "B사"], "current_index": 1, "company": {"name": "B사"}, "report_version": 2}
    assert guarded("market", boom)(state)["market_analysis"]["failed"] is True
    ex = guarded("explorer", boom)(state)
    assert ex["company"] == {"name": "B사"} and ex["is_eligible"] is False and ex["rejected"][0]["company"] == "B사"
    assert guarded("judge", boom)(state)["scores"]["decision"] == "보류"
    vr = guarded("verifier", boom)(state)["verify_result"]
    assert vr["finalized"] is True and vr["report_version"] == 2
    er = guarded("quality_eval", boom)(state)["eval_result"]
    assert er["error"] and er["report_version"] == 2


def test_reporter_exception_is_not_swallowed():
    with pytest.raises(TimeoutError):
        guarded("reporter", boom)({})


def test_quality_error_ends_without_rewrite():
    state = {"report": "r", "report_version": 1, "verify_result": {"report_version": 1, "passed": True},
             "eval_result": {"report_version": 1, "passed": False, "error": "RateLimit", "items": {}}}
    nxt, why, _ = supervisor.decide(state, 1)
    assert nxt == END and "오류" in why


# ---------- 그래프 안에서 ----------
def test_failing_analysis_is_retried_once_then_judged():
    s = Scenario(["A사"], {"A사": "투자"})
    nodes = s.nodes()
    calls = []

    def broken_team(state):
        calls.append("team")
        raise ConnectionError("Tavily 432")

    nodes["team"] = broken_team
    out = build_graph(nodes).invoke({"trace_id": "t"}, {"recursion_limit": 200})
    assert calls == ["team", "team"]                       # 첫 실행 + 재작업 1회
    assert out["team_analysis"]["failed"] is True
    assert out["evidence_status"]["team"]["sufficient"] is False
    assert len(out["agent_errors"]) == 2
    assert out["report"]                                    # 그래도 보고서까지 간다
