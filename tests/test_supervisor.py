"""Supervisor 라우팅 규칙 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_supervisor.py -q"""
import pytest
from langgraph.graph import END

import obs
from agents import supervisor as sup
from config import MAX_QUALITY_LOOPS, MAX_REPORT_RETRY, MAX_REWORK, MAX_STEPS
from state import CONTROL_KEYS


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "LOG_DIR", tmp_path / "decisions")


def src(node, publisher, company="A사"):
    return {"company": company, "node": node, "publisher": publisher, "url": f"https://{publisher}.com/x"}


def good_sources(company="A사"):
    return [src(n, p, company) for n in sup.ANALYSIS for p in ("pubA", "pubB")]


def analyzed(**over):
    """조건을 충족하고 4개 관점이 모두 충분한 기업 상태."""
    state = {
        "candidates": ["A사", "B사"], "current_index": 0,
        "company": {"name": "A사"}, "is_eligible": True,
        "tech_summary": {"missing_fields": ["weaknesses"]},
        "market_analysis": {"info_insufficient": False},
        "competitor_analysis": {"competitors": [{"name": "C사"}]},
        "team_analysis": {"members": [{"name": "홍길동"}], "concerns": []},
        "sources": good_sources(),
    }
    state.update(over)
    return state


def decide(state, step=1):
    return sup.decide(state, step)


# ---------- 후보 단계 ----------
def test_start_goes_to_explorer():
    nxt, _, upd = decide({})
    assert nxt == ["explorer"]
    assert upd["current_index"] == 0


def test_eligible_company_without_analysis_fans_out_to_four():
    nxt, why, _ = decide({"candidates": ["A사"], "current_index": 0, "company": {"name": "A사"}, "is_eligible": True})
    assert sorted(nxt) == sorted(sup.ANALYSIS)
    assert "미수집" in why


def test_only_missing_perspective_is_dispatched():
    nxt, _, _ = decide(analyzed(market_analysis=None))
    assert nxt == ["market"]


def test_ineligible_company_moves_to_next_candidate_and_resets_control():
    state = analyzed(is_eligible=False, eligibility_reason="상장사", rework_count={"market": 1})
    nxt, why, upd = decide(state)
    assert nxt == ["explorer"]
    assert upd["current_index"] == 1
    assert upd["rework_count"] == {} and upd["rework_request"] == {} and upd["evidence_status"] == {}
    assert "조건 미충족" in why and "B사" in why


def test_hold_moves_to_next_candidate():
    nxt, why, upd = decide(analyzed(scores={"decision": "보류", "total": 65}))
    assert nxt == ["explorer"] and upd["current_index"] == 1
    assert "보류" in why


def test_hold_on_last_candidate_writes_no_recommendation_report():
    nxt, why, upd = decide(analyzed(current_index=1, company={"name": "B사"}, scores={"decision": "보류", "total": 60}))
    assert nxt == ["reporter"]
    assert upd["report_version"] == 1
    assert "추천 기업 없음" in why


def test_no_candidates_found_writes_no_recommendation_report():
    nxt, _, _ = decide({"candidates": [], "current_index": 0, "company": None, "is_eligible": False})
    assert nxt == ["reporter"]


def test_invest_goes_to_reporter():
    nxt, _, upd = decide(analyzed(scores={"decision": "투자", "total": 78}))
    assert nxt == ["reporter"] and upd["report_version"] == 1


# ---------- 근거 충분성과 재작업 ----------
def test_sufficient_evidence_goes_to_judge():
    nxt, why, upd = decide(analyzed())
    assert nxt == ["judge"]
    assert "충분" in why
    assert all(s["sufficient"] for s in upd["evidence_status"].values())


def test_tech_missing_required_field_requests_rework_with_focus():
    state = analyzed(tech_summary={"missing_fields": ["development_stage", "performance_metrics", "weaknesses"]})
    nxt, why, upd = decide(state)
    assert nxt == ["tech_summary"]
    assert upd["rework_count"] == {"tech_summary": 1}
    assert upd["rework_request"]["tech_summary"]["focus"] == ["development_stage", "performance_metrics"]
    assert "개발 단계" in why and f"재작업 1/{MAX_REWORK}" in why


def test_only_insufficient_perspectives_are_reworked_in_parallel():
    state = analyzed(market_analysis={"info_insufficient": True}, team_analysis={"members": [], "concerns": []})
    nxt, _, upd = decide(state)
    assert sorted(nxt) == ["market", "team"]
    assert upd["rework_request"]["market"]["focus"] == ["upper_segment"]
    assert upd["rework_request"]["team"]["focus"] == ["alternative_queries"]


def test_competitor_without_competitors_is_insufficient():
    nxt, _, upd = decide(analyzed(competitor_analysis={"competitors": []}))
    assert nxt == ["competitor"]
    assert upd["rework_request"]["competitor"]["focus"] == ["english_keywords"]


def test_single_publisher_is_insufficient_for_bias_control():
    sources = [s for s in good_sources() if not (s["node"] == "market" and s["publisher"] == "pubB")]
    nxt, why, upd = decide(analyzed(sources=sources))
    assert nxt == ["market"]
    assert "발행처 1곳" in why
    assert "publisher_diversity" in upd["rework_request"]["market"]["focus"]


def test_sources_of_other_company_do_not_count():
    sources = [s for s in good_sources() if s["node"] != "team"] + [src("team", "pubA", "B사"), src("team", "pubB", "B사")]
    nxt, _, _ = decide(analyzed(sources=sources))
    assert nxt == ["team"]


def test_publisher_falls_back_to_domain():
    sources = [s for s in good_sources() if s["node"] != "team"] + [
        {"company": "A사", "node": "team", "publisher": "", "url": "https://www.news1.kr/a"},
        {"company": "A사", "node": "team", "publisher": "", "url": "https://zdnet.co.kr/b"},
    ]
    nxt, _, _ = decide(analyzed(sources=sources))
    assert nxt == ["judge"]


def test_rework_limit_reached_goes_to_judge_with_reason():
    state = analyzed(tech_summary={"missing_fields": ["development_stage"]}, rework_count={"tech_summary": MAX_REWORK})
    nxt, why, upd = decide(state)
    assert nxt == ["judge"]
    assert "한도 도달" in why
    assert upd["evidence_status"]["tech_summary"]["sufficient"] is False


def test_failed_agent_is_retried_once():
    nxt, why, upd = decide(analyzed(team_analysis={"failed": True, "error": "Timeout"}))
    assert nxt == ["team"]
    assert "실행 오류" in why
    assert upd["rework_request"]["team"]["focus"] == ["retry"]


# ---------- 보고서 단계 ----------
def report_state(**over):
    state = analyzed(scores={"decision": "투자", "total": 78}, report="# 보고서", report_version=1)
    state.update(over)
    return state


def test_new_report_goes_to_verifier():
    nxt, _, _ = decide(report_state())
    assert nxt == ["verifier"]


def test_verify_mismatch_rewrites_once():
    state = report_state(verify_result={"report_version": 1, "passed": False, "mismatches": [1, 2]})
    nxt, why, upd = decide(state)
    assert nxt == ["reporter"]
    assert upd == {"retry_count": 1, "report_version": 2}
    assert "불일치 2건" in why


def test_verify_mismatch_after_retry_limit_goes_to_quality():
    state = report_state(verify_result={"report_version": 1, "passed": False, "finalized": True},
                         retry_count=MAX_REPORT_RETRY)
    nxt, _, _ = decide(state)
    assert nxt == ["quality_eval"]


def test_verified_report_goes_to_quality_eval():
    nxt, _, _ = decide(report_state(verify_result={"report_version": 1, "passed": True}))
    assert nxt == ["quality_eval"]


def test_quality_fail_rewrites_and_bumps_version():
    state = report_state(verify_result={"report_version": 1, "passed": True},
                         eval_result={"report_version": 1, "passed": False,
                                      "items": {"groundedness": {"passed": False}, "neutrality": {"passed": True}}})
    nxt, why, upd = decide(state)
    assert nxt == ["reporter"]
    assert upd == {"quality_loops": 1, "report_version": 2}
    assert "groundedness" in why


def test_rewritten_report_is_verified_again_not_judged_by_old_results():
    # 재작성된 v2에 대해 이전 v1의 검증·평가 결과가 남아 있어도 다시 검증부터 한다
    state = report_state(report_version=2, verify_result={"report_version": 1, "passed": True},
                         eval_result={"report_version": 1, "passed": False, "items": {}}, quality_loops=1)
    nxt, _, _ = decide(state)
    assert nxt == ["verifier"]


def test_quality_pass_ends():
    state = report_state(verify_result={"report_version": 1, "passed": True},
                         eval_result={"report_version": 1, "passed": True, "items": {}})
    nxt, _, _ = decide(state)
    assert nxt == END


def test_quality_fail_at_loop_limit_ends():
    state = report_state(verify_result={"report_version": 1, "passed": True}, quality_loops=MAX_QUALITY_LOOPS,
                         eval_result={"report_version": 1, "passed": False, "items": {"bias_control": {"passed": False}}})
    nxt, why, _ = decide(state)
    assert nxt == END
    assert "한도 도달" in why


# ---------- 종료 가드 ----------
def test_step_limit_without_report_forces_report():
    nxt, _, upd = decide(analyzed(), step=MAX_STEPS + 1)
    assert nxt == ["reporter"]
    assert upd["report_version"] == 1 and upd["last_error"]


def test_step_limit_with_report_ends():
    nxt, _, _ = decide(report_state(), step=MAX_STEPS + 1)
    assert nxt == END


# ---------- 노드 래퍼 ----------
def test_node_returns_only_control_keys_and_logs_decision():
    out = sup.supervisor_node({**analyzed(), "trace_id": "t1", "step_count": 4})
    assert set(out) <= CONTROL_KEYS
    assert out["step_count"] == 5
    assert out["next_nodes"] == ["judge"]
    assert out["last_decision"].startswith("[5] judge")
    [event] = obs.read_events("t1", kind="decision")
    assert event["decision"] == ["judge"] and event["company"] == "A사" and event["step"] == 5


def test_route_reads_next_nodes_and_ends_when_empty():
    assert sup.route({"next_nodes": ["market", "team"]}) == ["market", "team"]
    assert sup.route({"next_nodes": []}) == END



def test_web_evidence_all_on_same_day_needs_independent_sources():
    same_day = [{"company": "A사", "node": "team", "kind": "web", "publisher": p, "url": f"https://{p}.kr",
                 "date": "2026-08-19"} for p in ("etnews", "mk")]
    sources = [s for s in good_sources() if s["node"] != "team"] + same_day
    nxt, why, upd = decide(analyzed(sources=sources))
    assert nxt == ["team"] and "같은 날(2026-08-19)" in why
    assert upd["rework_request"]["team"]["focus"] == ["publisher_diversity"]
    other_day = same_day[:1] + [{**same_day[1], "date": "2026-10-07"}]
    nxt, _, _ = decide(analyzed(sources=[s for s in good_sources() if s["node"] != "team"] + other_day))
    assert nxt == ["judge"]



# ---------- 근거 부족 → 관점 재조사 ----------
def failed_eval(gaps, version=1):
    return {"report_version": version, "passed": False, "items": {"bias_control": {"passed": False}},
            "evidence_gaps": gaps}


def test_quality_evidence_gap_routes_to_research():
    state = report_state(verify_result={"report_version": 1, "passed": True},
                         eval_result=failed_eval({"team": "같은 날 보도뿐"}))
    nxt, why, upd = decide(state)
    assert nxt == ["team"] and "관점 재조사 1/1" in why
    assert upd["reopen_stage"] == "research" and upd["research_loops"] == 1 and upd["quality_loops"] == 1
    assert upd["rework_request"]["team"]["reason"].startswith("품질 평가:")
    assert upd["rework_request"]["team"]["focus"] == sup.RESEARCH_FOCUS["team"]   # 충분성 판정은 통과했던 관점


def test_research_loop_limit_falls_back_to_rewrite():
    state = report_state(verify_result={"report_version": 1, "passed": True}, research_loops=1,
                         eval_result=failed_eval({"team": "같은 날 보도뿐"}))
    nxt, _, upd = decide(state)
    assert nxt == ["reporter"] and upd == {"quality_loops": 1, "report_version": 2}


def test_reopen_research_then_rejudge():
    nxt, _, upd = decide(report_state(reopen_stage="research"))
    assert nxt == ["judge"] and upd["reopen_stage"] == "judge" and upd["rework_request"] == {}


def test_reopen_rejudge_invest_rewrites_report():
    nxt, why, upd = decide(report_state(reopen_stage="judge", report_version=1))
    assert nxt == ["reporter"] and upd == {"reopen_stage": None, "report_version": 2}
    assert "재판단 투자" in why


def test_reopen_rejudge_hold_moves_on_or_writes_no_recommendation():
    nxt, why, upd = decide(report_state(reopen_stage="judge", scores={"decision": "보류", "total": 62}))
    assert nxt == ["explorer"] and upd["current_index"] == 1 and upd["reopen_stage"] is None
    assert upd["research_loops"] == 0 and upd["quality_loops"] == 0
    last = report_state(reopen_stage="judge", scores={"decision": "보류", "total": 62}, current_index=1,
                        company={"name": "B사"})
    nxt, why, _ = decide(last)
    assert nxt == ["reporter"] and "추천 기업 없음" in why
