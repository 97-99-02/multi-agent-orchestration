"""Supervisor 패턴 그래프.

모든 하위 노드는 실행 후 supervisor로만 돌아간다 (하위 노드끼리 직접 연결 없음).
다음에 무엇을 실행할지는 supervisor가 State를 보고 정하고, 분기는 add_conditional_edges 하나로 한다.
supervisor가 노드 여러 개를 돌려주면 같은 단계에서 병렬로 실행되고, 모두 끝난 뒤 supervisor가 한 번 실행된다.
"""
from langgraph.graph import END, START, StateGraph

from agents.competitor import competitor_node
from agents.explorer import explorer_node
from agents.judge import judge_node
from agents.market import market_node
from agents.quality import quality_eval_node
from agents.reporter import reporter_node
from agents.supervisor import route, supervisor_node
from agents.team import team_node
from agents.tech_summary import tech_summary_node
from agents.verifier import verifier_node
from state import GraphState

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


def build_graph(nodes: dict | None = None, checkpointer=None):
    """nodes로 하위 노드를 바꿔 끼울 수 있다 (테스트에서 가짜 노드로 흐름만 확인할 때)."""
    workers = {**WORKERS, **(nodes or {})}
    g = StateGraph(GraphState)
    g.add_node("supervisor", supervisor_node)
    for name, fn in workers.items():
        g.add_node(name, fn)
        g.add_edge(name, "supervisor")
    g.add_edge(START, "supervisor")
    g.add_conditional_edges("supervisor", route, [*workers, END])
    return g.compile(checkpointer=checkpointer)
