"""Supervisor 재작업 입력 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_rework.py -q

에이전트는 검색어와 seed가 고정이라 같은 검색어로 다시 돌리면 같은 결과만 나온다. 재작업 때는
1) 첫 실행(노드 안 재검색 포함)과 다른 검색어를 쓰고 2) 이전에 확인한 사실은 유지하는지 확인한다.
"""
from types import SimpleNamespace

from langchain_core.documents import Document

from agents import competitor, market, team
from agents import tech_summary as tech
from agents.rework import previous, rework_of, union


def rework_state(node, focus, **extra):
    return {"rework_request": {node: {"reason": "부족", "focus": focus}}, **extra}


# ---------- 공통 도우미 ----------
def test_rework_of_ignores_other_nodes_and_retry_only():
    assert rework_of({}, "team") is None
    assert rework_of(rework_state("market", ["upper_segment"]), "team") is None
    assert rework_of(rework_state("team", ["retry"]), "team") is None        # 실행 오류 재시도는 첫 실행과 같게
    assert rework_of(rework_state("team", ["alternative_queries"]), "team")["focus"] == ["alternative_queries"]


def test_previous_skips_failed_result():
    assert previous({"team_analysis": {"failed": True}}, "team_analysis") is None
    assert previous({"team_analysis": {"members": []}}, "team_analysis") == {"members": []}


def test_union_keeps_order_and_old_value():
    assert union(["a", "b"], ["b", "c"]) == ["a", "b", "c"]


# ---------- 기술 요약 ----------
def test_tech_rework_queries_differ_from_first_run_and_node_retry():
    first = tech._queries("A사", "엣지", list(tech.FIELDS)) + tech._queries("A사", "엣지", list(tech.FIELDS), retry=True)
    general, news = tech._rework_queries("A사", "X1", ["development_stage", "performance_metrics"])
    assert general and not set(general) & set(first)
    assert all("X1" in q for q in general)
    assert news == []
    assert tech._rework_queries("A사", "X1", ["publisher_diversity"])[1]   # 발행처 편중이면 뉴스 검색 추가


def test_tech_node_rework_uses_rework_queries_and_keeps_previous_facts(monkeypatch):
    calls = []
    monkeypatch.setattr(tech, "_collect_web", lambda queries, topic="general": calls.append((tuple(queries), topic)) or [])
    monkeypatch.setattr(tech, "_relevant", lambda name, items: ([], ""))
    monkeypatch.setattr(tech, "_grounded", lambda name, items, audit=None: [])
    monkeypatch.setattr(tech, "_industry_context", lambda *a: ([], [], "no_company_facts", {}))
    prev = {"core_chips": ["X1 NPU"], "development_stage": [], "evidence": [{"field": "core_chips", "text": "X1 NPU"}],
            "industry_context": [{"text": "업계 기준"}], "industry_context_status": "ok"}
    for f in tech.FIELDS:
        prev.setdefault(f, [])
    state = rework_state("tech_summary", ["development_stage"],
                         company={"name": "A사", "segment": "엣지", "product": "X1"}, tech_summary=prev)
    out = tech.tech_summary_node(state)["tech_summary"]
    assert len(calls) == 1 and "tape-out" in calls[0][0][0]      # 재작업 검색 1번, 노드 안 재검색 없음
    assert out["core_chips"] == ["X1 NPU"]                         # 이전에 확인한 사실 유지
    assert "core_chips" not in out["missing_fields"] and "development_stage" in out["missing_fields"]
    assert out["industry_context"] == [{"text": "업계 기준"}]
    assert out["diagnostics"]["rework"]["focus"] == ["development_stage"]


def test_tech_merge_adds_new_facts_without_duplicates():
    prev = {f: [] for f in tech.FIELDS} | {"core_chips": ["X1 NPU"], "evidence": [{"field": "core_chips", "text": "X1 NPU"}]}
    new = {f: [] for f in tech.FIELDS} | {"core_chips": ["X1  NPU"], "development_stage": ["2025년 테이프아웃"],
                                          "evidence": [{"field": "development_stage", "text": "2025년 테이프아웃"}]}
    merged = tech._merge_previous(prev, new)
    assert merged["core_chips"] == ["X1 NPU"]                     # 공백만 다른 같은 문장은 하나로
    assert merged["development_stage"] == ["2025년 테이프아웃"]
    assert len(merged["evidence"]) == 2
    assert merged["field_status"]["development_stage"] == "supported"


# ---------- 시장성 ----------
def fake_extract(figures, summary="요약"):
    analysis = SimpleNamespace(demand_drivers=["AI PC 확산"], market_risks=["공급 과잉"], summary=summary, evidence_ids=[])
    return lambda name, segment, docs: (analysis, [dict(f) for f in figures])


def test_market_rework_searches_other_queries_and_both_doc_types(monkeypatch):
    calls = []
    monkeypatch.setattr(market, "retrieve", lambda queries, doc_type: calls.append((tuple(queries), doc_type)) or [])
    monkeypatch.setattr(market, "extract", fake_extract([]))
    market.market_node(rework_state("market", ["upper_segment"], company={"name": "A사", "segment": "엣지"}))
    assert [d for _, d in calls] == ["market", "tech"]
    used = set(calls[0][0])
    assert not used & set(market.SEGMENT_QUERIES["엣지"])           # 첫 실행 검색어와 다름
    assert used & set(market.SEGMENT_QUERIES["기타"])               # 상위 분야 검색어 포함


def test_market_rework_keeps_previous_figures(monkeypatch):
    doc = Document(page_content="온디바이스 AI 반도체 시장은 연평균 25% 성장", metadata={"title": "시장 보고서", "page": 3})
    monkeypatch.setattr(market, "retrieve", lambda queries, doc_type: [doc] if doc_type == "market" else [])
    new_fig = {"kind": "성장률", "value": "25%", "year": "2024~2030", "scope": "온디바이스 AI 반도체",
               "segment_specific": True, "evidence_id": 0}
    monkeypatch.setattr(market, "extract", fake_extract([new_fig], summary="정보 부족"))
    old_fig = {"kind": "시장규모", "value": "100억 달러", "year": "2030", "scope": "온디바이스 AI 반도체", "segment_specific": True}
    prev = {"figures": [old_fig], "demand_drivers": ["AI PC 확산"], "market_risks": [], "summary": "이전 요약",
            "searched": ["market", "tech"], "info_insufficient": True}
    out = market.market_node(rework_state("market", ["upper_segment"], company={"name": "A사", "segment": "엣지"},
                                          market_analysis=prev))["market_analysis"]
    assert [f["kind"] for f in out["figures"]] == ["시장규모", "성장률"]
    assert out["info_insufficient"] is False                        # 합친 수치로 다시 판정
    assert out["demand_drivers"] == ["AI PC 확산"]
    assert out["summary"] == "이전 요약"
    assert "rework" in out["searched"]


def test_market_first_run_unchanged(monkeypatch):
    calls = []
    monkeypatch.setattr(market, "retrieve", lambda queries, doc_type: calls.append((tuple(queries), doc_type)) or [])
    monkeypatch.setattr(market, "extract", fake_extract([]))
    market.market_node({"company": {"name": "A사", "segment": "엣지"}})
    assert calls == [(tuple(market.SEGMENT_QUERIES["엣지"]), "market"), (tuple(market.SEGMENT_QUERIES["엣지"]), "tech")]


# ---------- 팀 ----------
class FakeTeamLLM:
    def __init__(self, members):
        self.members = members

    def with_structured_output(self, _):
        return self

    def with_retry(self, **_):
        return self

    def invoke(self, prompt):
        return team.TeamAnalysis(members=[team.Member(name=n, role="CTO", background="KAIST") for n in self.members],
                                 team_size="", strengths=["반도체 설계 경험"], concerns=[], evidence_ids=[0])


def run_team(monkeypatch, state, members):
    queries = []

    def fake_search(qs, max_results=5, topic="news"):
        queries.append((tuple(qs), topic))
        return [{"title": "A사 인터뷰", "content": "A사 CTO", "url": "https://news.kr/1"}]

    monkeypatch.setattr(team, "search_many", fake_search)
    monkeypatch.setattr(team, "ChatOpenAI", lambda **k: FakeTeamLLM(members))
    return team.team_node(state)["team_analysis"], queries


def test_team_rework_uses_other_queries_and_merges_members(monkeypatch):
    prev = {"members": [{"name": "홍길동", "role": "대표", "background": "삼성전자"}], "team_size": "30명",
            "strengths": [], "concerns": ["공개 정보 부족"]}
    state = rework_state("team", ["alternative_queries", "publisher_diversity"],
                         company={"name": "A사", "ceo": "홍길동"}, team_analysis=prev)
    out, queries = run_team(monkeypatch, state, ["김철수"])
    first = {"A사 홍길동 대표 경력", "A사 창업자 CTO 경력", "A사 대표 인터뷰"}
    assert not {q for qs, _ in queries for q in qs} & first
    assert [t for _, t in queries] == ["general", "news"]
    assert [m["name"] for m in out["members"]] == ["홍길동", "김철수"]
    assert "공개 정보 부족" not in out["concerns"]                   # 인력을 찾았으니 우려에서 뺀다
    assert out["team_size"] == "30명"


def test_team_first_run_unchanged(monkeypatch):
    out, queries = run_team(monkeypatch, {"company": {"name": "A사", "ceo": "홍길동"}}, ["홍길동"])
    assert queries == [(("A사 홍길동 대표 경력", "A사 창업자 CTO 경력", "A사 대표 인터뷰"), "general")]


# ---------- 경쟁사 ----------
def test_competitor_rework_queries_differ_from_first_run():
    first = {"A사 경쟁사 비교", "엣지 AI NPU 시장 주요 업체", "온디바이스 엣지 AI NPU 스타트업 투자 유치",
             competitor.SEGMENT_KEYWORDS_EN["엣지"]}
    general, news = competitor.rework_queries("A사", "X1", "엣지", ["english_keywords"])
    assert not (set(general) | set(news)) & first
    assert any("X1" in q for q in general)
    assert len(competitor.rework_queries("A사", "X1", "엣지", ["publisher_diversity"])[1]) == 2


def test_competitor_merge_keeps_previous_and_drops_not_found_risk():
    prev = {"competitors": [], "differentiation": ["SDK"], "competitive_risks": [competitor.NOT_FOUND]}
    new = {"competitors": [{"name": "딥엑스 (DEEPX)"}], "differentiation": [], "competitive_risks": ["가격 경쟁"]}
    merged = competitor._merge_previous(prev, new)
    assert [c["name"] for c in merged["competitors"]] == ["딥엑스 (DEEPX)"]
    assert merged["competitive_risks"] == ["가격 경쟁"]
    assert merged["differentiation"] == ["SDK"]
