"""그래프 State 정의 (Supervisor 패턴).

설계 원칙
- State에는 제어 흐름과 재개에 필요한 것만 둔다. 결정 로그·진단 본문은 obs.py로 외부에 남긴다.
- 페이로드(작업 결과)는 하위 에이전트가 쓰고, 제어 메타(라우팅·종료·재개)는 supervisor만 쓴다.
- trace_id 하나로 State, LangSmith 트레이스, 결정 로그, 체크포인터 thread_id를 잇는다.
"""
import operator
from typing import Annotated, Optional, TypedDict


class Source(TypedDict, total=False):
    """sources 리스트의 항목 형식. 모든 에이전트가 같은 형식으로 기록한다."""
    company: str        # 어느 기업 분석에 쓴 출처인지 (REFERENCE는 투자 기업 것만 사용)
    node: str           # 기록한 에이전트 이름 (explorer, tech_summary, market, competitor, team)
    kind: str           # "web" | "report" (RAG 문서)
    title: str          # 기사 제목 또는 보고서명
    publisher: str      # 언론사·발행기관
    date: str           # YYYY-MM-DD 또는 YYYY
    url: str
    snippet: str        # 근거로 쓴 원문 일부 (사실 검증에 사용)
    page: int           # RAG 문서 출처의 쪽 번호 (웹 출처는 없음)
    source_id: str      # 기술 요약 주장과 이 출처를 연결하는 키
    source_file: str    # 청크가 나온 원본 문서 파일명
    source_path: str    # 프로젝트 루트 기준 원본 파일 위치 (있을 때)


def _source_key(s: Source) -> tuple:
    return (s.get("company"), s.get("node"), s.get("source_id"), s.get("url"), s.get("page"), s.get("snippet"))


def merge_sources(old: list[Source], new: list[Source]) -> list[Source]:
    """재작업으로 같은 에이전트가 같은 출처를 다시 넣어도 한 번만 남긴다.
    다른 에이전트가 같은 기사를 쓴 경우(node가 다름)는 지금처럼 둘 다 남긴다.
    """
    seen = {_source_key(s) for s in old}
    merged = list(old)
    for s in new:
        key = _source_key(s)
        if key not in seen:
            seen.add(key)
            merged.append(s)
    return merged


class GraphState(TypedDict, total=False):
    # ── 페이로드: 하위 에이전트가 쓴다 ─────────────────────────────
    candidates: list[str]                 # 탐색 에이전트가 발굴한 후보 (최대 5개)
    company: Optional[dict]               # 기업명, 세부 분야, 투자 단계, 대표자, 투자 금액·기업가치
    is_eligible: bool                     # 상세 확인 결과 평가 조건 충족 여부
    eligibility_reason: str

    # 분석 노드 4개는 병렬로 돌 수 있으므로 각자 다른 키에만 기록한다
    tech_summary: Optional[dict]
    market_analysis: Optional[dict]
    competitor_analysis: Optional[dict]
    team_analysis: Optional[dict]

    scores: Optional[dict]   # 항목별 점수·근거·출처, legal_risk, total, decision, risks
    report: str
    verify_result: dict      # {passed, mismatches, ..., report_version}
    eval_result: Optional[dict]  # 품질 평가 {passed, report_version, items, fix_instructions}

    # 여러 노드가 같은 단계에 쓰므로 reducer로 병합한다 (동시 처리)
    sources: Annotated[list[Source], merge_sources]
    rejected: Annotated[list[dict], operator.add]      # {company, reason, total}
    agent_errors: Annotated[list[dict], operator.add]  # {company, node, error}. supervisor가 재시도·진행을 정한다

    # ── 제어 메타: supervisor만 쓴다 ───────────────────────────────
    trace_id: str                          # 외부 결정 로그·LangSmith metadata와 잇는 키
    step_count: int                        # supervisor 실행 횟수 (종료 가드)
    current_index: int                     # 다음에 평가할 후보 순번
    next_nodes: list[str]                  # 이번 라우팅 결과
    evidence_status: dict[str, dict]       # {관점: {"sufficient": bool, "reason": str}}
    rework_count: dict[str, int]           # 관점별 재작업 횟수 (현재 기업 기준)
    rework_request: dict[str, dict]        # {관점: {"reason": str, "focus": [...]}}. 해당 에이전트가 읽는다
    report_version: int                    # reporter가 새로 쓸 때마다 +1 (verifier의 메모 추가는 해당 없음)
    retry_count: int                       # 사실 검증 불일치로 인한 재작성 횟수
    quality_loops: int                     # 품질 평가 미달로 인한 재작성 횟수
    last_decision: str                     # 마지막 라우팅 결정 한 줄 (상세는 obs.py 결정 로그)
    last_error: Optional[str]


# supervisor만 쓰는 키. 하위 에이전트 반환값에 이 키가 있으면 안 된다 (테스트로 확인)
CONTROL_KEYS = frozenset({
    "trace_id", "step_count", "current_index", "next_nodes", "evidence_status", "rework_count",
    "rework_request", "report_version", "retry_count", "quality_loops", "last_decision", "last_error",
})

# 다음 후보로 넘어갈 때 탐색 에이전트가 비우는 페이로드 (이전 기업 내용이 섞이지 않게)
RESET_ON_NEXT = {
    "company": None,
    "is_eligible": False,
    "eligibility_reason": "",
    "tech_summary": None,
    "market_analysis": None,
    "competitor_analysis": None,
    "team_analysis": None,
    "scores": None,
}

# 다음 후보로 넘어갈 때 supervisor가 비우는 제어 메타 (재작업 횟수·요청은 기업 단위)
CONTROL_RESET_ON_NEXT = {
    "evidence_status": {},
    "rework_count": {},
    "rework_request": {},
}
