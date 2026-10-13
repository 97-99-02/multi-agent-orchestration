"""State 설계 규칙 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_state.py -q"""
from typing import get_type_hints

from langgraph.graph import END, START, StateGraph

from state import CONTROL_KEYS, CONTROL_RESET_ON_NEXT, RESET_ON_NEXT, GraphState, merge_sources

FIELDS = set(get_type_hints(GraphState))


def src(node="market", url="https://a.com", snippet="시장 규모 10조", company="A사", **kw):
    return {"company": company, "node": node, "url": url, "snippet": snippet, **kw}


def test_merge_sources_drops_same_source_from_same_node():
    # 재작업으로 같은 에이전트가 같은 출처를 다시 넣는 경우
    merged = merge_sources([src()], [src(), src(url="https://b.com")])
    assert [s["url"] for s in merged] == ["https://a.com", "https://b.com"]


def test_merge_sources_keeps_same_article_from_different_nodes():
    merged = merge_sources([src(node="market")], [src(node="team")])
    assert [s["node"] for s in merged] == ["market", "team"]


def test_merge_sources_keeps_different_pages_of_same_document():
    merged = merge_sources([src(url="", page=3)], [src(url="", page=4)])
    assert len(merged) == 2


def test_control_and_reset_keys_are_declared_in_state():
    assert CONTROL_KEYS <= FIELDS
    assert set(RESET_ON_NEXT) <= FIELDS
    assert set(CONTROL_RESET_ON_NEXT) <= CONTROL_KEYS


def test_payload_reset_does_not_touch_control_keys():
    # 하위 에이전트(탐색)가 쓰는 초기화 목록에 제어 키가 섞이면 '제어 키는 supervisor만' 규칙이 깨진다
    assert not set(RESET_ON_NEXT) & CONTROL_KEYS


def test_parallel_nodes_merge_through_reducers():
    # 같은 단계에서 두 노드가 sources·agent_errors에 동시에 써도 InvalidUpdateError 없이 병합된다
    def a(state):
        return {"sources": [src(node="market")], "agent_errors": [{"company": "A사", "node": "market", "error": "x"}]}

    def b(state):
        return {"sources": [src(node="market"), src(node="team")],
                "agent_errors": [{"company": "A사", "node": "team", "error": "y"}]}

    g = StateGraph(GraphState)
    g.add_node("a", a)
    g.add_node("b", b)
    g.add_edge(START, "a")
    g.add_edge(START, "b")
    g.add_edge(["a", "b"], END)
    out = g.compile().invoke({})
    assert [s["node"] for s in out["sources"]] == ["market", "team"]
    assert {e["node"] for e in out["agent_errors"]} == {"market", "team"}


def test_sources_reset_marker_drops_previous_company_sources():
    from state import SOURCES_RESET
    old = [src(company="A사"), src(company="A사", url="https://b.com")]
    merged = merge_sources(old, [SOURCES_RESET, src(company="B사")])
    assert [s["company"] for s in merged] == ["B사"]
    assert SOURCES_RESET not in merged


def test_explorer_resets_sources_when_moving_to_next_candidate(monkeypatch):
    from agents import explorer
    from state import SOURCES_RESET
    monkeypatch.setattr(explorer, "verify_company", lambda name: {
        "company": {"name": name}, "is_eligible": True, "eligibility_reason": "",
        "sources": [src(company=name, node="explorer")]})
    out = explorer.explorer_node({"candidates": ["A사", "B사"], "current_index": 1})
    assert out["sources"][0] == SOURCES_RESET and out["sources"][1]["company"] == "B사"
