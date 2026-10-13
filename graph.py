"""Supervisor 패턴 그래프.

모든 하위 노드는 실행 후 supervisor로만 돌아간다 (하위 노드끼리 직접 연결 없음).
다음에 무엇을 실행할지는 supervisor가 State를 보고 정하고, 분기는 add_conditional_edges 하나로 한다.
supervisor가 노드 여러 개를 돌려주면 같은 단계에서 병렬로 실행되고, 모두 끝난 뒤 supervisor가 한 번 실행된다.

하위 노드 예외는 guarded()가 잡아 노드별 대체 결과와 agent_errors로 바꾼다. 전체 실행이 죽지 않고,
supervisor가 대체 결과를 보고 재시도(분석 노드) 또는 다음 단계 진행을 정한다.
"""
import functools
import logging

from langgraph.graph import END, START, StateGraph

from agents.competitor import competitor_node
from agents.explorer import explorer_node
from agents.judge import judge_node
from agents.market import market_node
from agents.quality import quality_eval_node
from agents.reporter import reporter_node
from agents.supervisor import ANALYSIS, route, supervisor_node
from agents.team import team_node
from agents.tech_summary import tech_summary_node
from agents.verifier import verifier_node
from obs import log_event
from state import RESET_ON_NEXT, GraphState

logger = logging.getLogger(__name__)

WORKERS = {
    "explorer": explorer_node,
    "tech_summary": tech_summary_node,
    "market": market_node,
    "competitor": competitor_node,
    "team": team_node,
    "judge": judge_node,
    "reporter": reporter_node,
    "verifier": verifier_node,
    "quality_eval": quality_eval_node,
}


def fallback(name: str, state: dict, error: str) -> dict:
    """노드가 예외로 끝났을 때 State에 남길 대체 결과. supervisor가 이 값을 보고 다음을 정한다."""
    company = (state.get("company") or {}).get("name")
    if name in ANALYSIS:   # 분석: 실패 표시 → supervisor가 재작업 1회, 그래도 실패면 judge가 정보 부족(최대 2점)
        return {ANALYSIS[name]: {"failed": True, "error": error}}
    if name == "explorer":   # 탐색: 이 후보는 조건 미충족으로 보고 다음 후보로
        candidates = state.get("candidates") or []
        idx = state.get("current_index", 0)
        target = candidates[idx] if idx < len(candidates) else None
        out = {**RESET_ON_NEXT, "candidates": candidates, "company": {"name": target} if target else None,
               "is_eligible": False, "eligibility_reason": f"탐색 오류: {error}"}
        if target:
            out["rejected"] = [{"company": target, "reason": f"탐색 오류: {error}", "total": None}]
        return out
    if name == "judge":   # 판단: 보류로 보고 다음 후보로
        return {"scores": {"decision": "보류", "total": None, "hold_reasons": [f"판단 오류: {error}"], "items": {}},
                "rejected": [{"company": company, "reason": f"판단 오류: {error}", "total": None}]}
    if name == "verifier":   # 사실 검증: 검증 못 한 채 확정하고 품질 평가로
        return {"verify_result": {"passed": False, "finalized": True, "mismatches": [], "error": error,
                                  "report_version": state.get("report_version")}}
    if name == "quality_eval":   # 품질 평가: 오류 표시 → supervisor가 종료
        return {"eval_result": {"passed": False, "error": error, "items": {}, "fix_instructions": "",
                                "report_version": state.get("report_version")}}
    raise KeyError(name)


def guarded(name: str, fn):
    """reporter는 대체할 결과가 없어 예외를 그대로 올린다. 나머지 노드는 대체 결과로 바꾼다."""
    if name == "reporter":
        return fn

    @functools.wraps(fn)
    def run(state):
        try:
            return fn(state)
        except Exception as e:  # noqa: BLE001 - 어떤 오류든 실행 전체를 멈추지 않는다
            error = f"{type(e).__name__}: {e}"[:300]
            company = (state.get("company") or {}).get("name")
            logger.exception("[%s] 실행 오류 → 대체 결과로 진행", name)
            log_event(state.get("trace_id"), "error", name, company=company, error=error)
            return {**fallback(name, state, error), "agent_errors": [{"company": company, "node": name, "error": error}]}
    return run


def build_graph(nodes: dict | None = None, checkpointer=None):
    """nodes로 하위 노드를 바꿔 끼울 수 있다 (테스트에서 가짜 노드로 흐름만 확인할 때)."""
    workers = {**WORKERS, **(nodes or {})}
    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor_node)
    for name, fn in workers.items():
        g.add_node(name, guarded(name, fn))
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route, [*workers, END])
    return g.compile(checkpointer=checkpointer)
