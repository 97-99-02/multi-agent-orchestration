"""Supervisor 노드 (담당: 이채목)

모든 하위 노드는 실행 후 supervisor로 돌아오고, supervisor가 현재 State를 보고 다음 노드를 정한다.
단계 번호로 분기하지 않고 "무엇이 State에 있고 무엇이 부족한가"로만 판단한다.

입력: State 전체 (읽기만)
출력: 제어 메타만 (step_count, next_nodes, last_decision, current_index, evidence_status,
      rework_count, rework_request, report_version, retry_count, quality_loops, last_error)
결정과 사유는 obs.log_decision으로 outputs/decisions/{trace_id}.jsonl에 남긴다.

라우팅 규칙 (위에서부터 처음 맞는 것)
 0. supervisor 실행 횟수가 MAX_STEPS 초과 → 보고서가 없으면 reporter, 있으면 종료
 1. 보고서가 있음 → 사실 검증 → (불일치면 재작성) → 품질 평가 → (미달이면 재작성) → 종료
 2. 현재 기업이 투자 결정 → reporter
 3. 아직 후보 발굴 전 → explorer
 4. 현재 기업이 조건 미충족 또는 보류 → 남은 후보가 있으면 explorer, 없으면 reporter("추천 기업 없음")
 5. 수집하지 않은 관점이 있음 → 그 관점들 (병렬)
 6. 근거 부족 관점이 있고 재작업 한도 미만 → 그 관점들 (병렬, rework_request에 사유·보완 대상)
 7. 그 외 → judge (한도에 걸린 관점은 정보 부족인 채로 진행, judge가 최대 2점으로 제한)
"""
from urllib.parse import urlparse

from langgraph.graph import END

from config import MAX_QUALITY_LOOPS, MAX_REPORT_RETRY, MAX_REWORK, MAX_STEPS
from obs import log_decision
from state import CONTROL_RESET_ON_NEXT

DECISION_INVEST = "투자"
DECISION_HOLD = "보류"

# 분석 노드 이름 → State 키
ANALYSIS = {
    "tech_summary": "tech_summary",
    "market": "market_analysis",
    "competitor": "competitor_analysis",
    "team": "team_analysis",
}

# 기술 요약에서 비면 근거 부족으로 보는 항목. 채점 기준표의 기술·제품 성숙도(개발 단계, 핵심 칩, 성능)와
# 실적·고객 검증(공개 매출·계약)에 직접 쓰인다. 강점·약점·공정은 비어도 채점 근거가 남아 제외한다
TECH_REQUIRED = {
    "core_chips": "핵심 칩",
    "development_stage": "개발 단계",
    "performance_metrics": "성능 수치",
    "public_revenue_contracts": "공개 매출·계약",
}

# 편향 통제: 한 관점의 근거가 발행처 1곳에만 기대면 부족으로 본다
MIN_PUBLISHERS = 2


def _publisher(s: dict) -> str:
    """발행처. 웹 출처는 제목에서 언론사를 못 뽑는 경우가 있어 도메인으로 대신한다."""
    if s.get("publisher"):
        return s["publisher"].strip()
    if s.get("url"):
        return urlparse(s["url"]).netloc.removeprefix("www.")
    return s.get("source_file") or s.get("title") or ""


def publishers_of(sources: list[dict], company: str, node: str) -> set[str]:
    found = {_publisher(s) for s in sources if s.get("company") == company and s.get("node") == node}
    found.discard("")
    return found


def check_evidence(node: str, analysis: dict, sources: list[dict], company: str) -> dict:
    """관점 하나의 근거 충분성. {"sufficient", "reason", "focus"}를 돌려준다.
    focus는 재작업 때 해당 에이전트가 무엇을 보완할지 읽는 값이다.
    """
    reasons, focus = [], []
    if analysis.get("failed"):
        reasons.append(f"실행 오류({analysis.get('error', '')[:60]})")
        focus.append("retry")
    elif node == "tech_summary":
        missing = [f for f in TECH_REQUIRED if f in (analysis.get("missing_fields") or [])]
        if missing:
            reasons.append("·".join(TECH_REQUIRED[f] for f in missing) + " 없음")
            focus += missing
    elif node == "market":
        if analysis.get("info_insufficient"):
            reasons.append("세부 분야 시장 규모·성장률 중 빠진 수치 있음")
            focus.append("upper_segment")
    elif node == "team":
        if not analysis.get("members") or any("공개 정보 부족" in c for c in analysis.get("concerns") or []):
            reasons.append("창업자·핵심 인력 공개 정보 부족")
            focus.append("alternative_queries")
    elif node == "competitor":
        if not analysis.get("competitors"):
            reasons.append("같은 세부 분야 경쟁사 없음")
            focus.append("english_keywords")

    if not analysis.get("failed"):
        pubs = publishers_of(sources, company, node)
        if len(pubs) < MIN_PUBLISHERS:
            reasons.append(f"근거 발행처 {len(pubs)}곳")
            focus.append("publisher_diversity")
    return {"sufficient": not reasons, "reason": ", ".join(reasons) or "충분", "focus": focus}


def evidence_report(state: dict) -> dict[str, dict]:
    company = (state.get("company") or {}).get("name", "")
    sources = state.get("sources") or []
    return {node: check_evidence(node, state.get(key) or {}, sources, company) for node, key in ANALYSIS.items()}


def _report_stage(state: dict, version: int) -> tuple:
    verify = state.get("verify_result") or {}
    if verify.get("report_version") != version:
        return ["verifier"], f"보고서 v{version} 사실 검증", {}

    retry = state.get("retry_count", 0)
    if not verify.get("passed") and not verify.get("finalized") and retry < MAX_REPORT_RETRY:
        n = len(verify.get("mismatches") or [])
        return (["reporter"], f"사실 검증 불일치 {n}건 → 재작성 {retry + 1}/{MAX_REPORT_RETRY}",
                {"retry_count": retry + 1, "report_version": version + 1})

    evaluation = state.get("eval_result") or {}
    if evaluation.get("report_version") != version:
        return ["quality_eval"], f"보고서 v{version} 품질 평가", {}

    loops = state.get("quality_loops", 0)
    if evaluation.get("passed"):
        return END, f"보고서 v{version} 품질 통과 → 종료", {}
    failed = [k for k, v in (evaluation.get("items") or {}).items() if not v.get("passed")]
    if loops < MAX_QUALITY_LOOPS:
        return (["reporter"], f"품질 미달({', '.join(failed)}) → 재작성 {loops + 1}/{MAX_QUALITY_LOOPS}",
                {"quality_loops": loops + 1, "report_version": version + 1})
    return END, f"품질 미달({', '.join(failed)})이나 Loop 한도 도달 → 미달 항목을 한계점에 남기고 종료", {}


def decide(state: dict, step: int) -> tuple:
    """(다음 노드 목록 또는 END, 사유, 갱신할 제어 메타)."""
    version = state.get("report_version", 0)
    report = state.get("report")

    # 0. 종료 가드
    if step > MAX_STEPS:
        halt = f"supervisor 실행 {MAX_STEPS}회 초과"
        if not report:
            return ["reporter"], f"{halt} → 지금까지 결과로 보고서 작성", {"report_version": version + 1, "last_error": halt}
        return END, f"{halt} → 종료", {"last_error": halt}

    # 1. 보고서 단계
    if report:
        return _report_stage(state, version)

    company = state.get("company")
    name = (company or {}).get("name", "")
    scores = state.get("scores") or {}
    candidates = state.get("candidates")
    idx = state.get("current_index", 0)

    # 2. 투자 결정
    if scores.get("decision") == DECISION_INVEST:
        return ["reporter"], f"{name} 투자 결정({scores.get('total')}점) → 보고서 작성", {"report_version": version + 1}

    # 3. 후보 발굴 전
    if candidates is None:
        return ["explorer"], "평가 중인 기업 없음 → 후보 발굴", {"current_index": idx}

    # 4. 현재 기업 평가 종료 (조건 미충족·보류) 또는 발굴 결과 없음 → 다음 후보
    if company is None or not state.get("is_eligible") or scores.get("decision") == DECISION_HOLD:
        if company is None:
            why, next_idx = "발굴 결과 없음", idx
        elif not state.get("is_eligible"):
            why, next_idx = f"{name} 조건 미충족({state.get('eligibility_reason', '')})", idx + 1
        else:
            why, next_idx = f"{name} 보류({scores.get('total')}점)", idx + 1
        if next_idx < len(candidates):
            return (["explorer"], f"{why} → 다음 후보 {candidates[next_idx]}",
                    {"current_index": next_idx, **CONTROL_RESET_ON_NEXT})
        return ["reporter"], f"{why}, 후보 소진 → '추천 기업 없음' 보고서", {"report_version": version + 1}

    # 5. 수집하지 않은 관점
    missing = [node for node, key in ANALYSIS.items() if state.get(key) is None]
    if missing:
        return missing, f"{name} 미수집 관점: {', '.join(missing)}", {}

    # 6. 근거 부족 관점 재작업
    status = evidence_report(state)
    counts = dict(state.get("rework_count") or {})
    rework = [n for n, s in status.items() if not s["sufficient"] and counts.get(n, 0) < MAX_REWORK]
    if rework:
        requests = {n: {"reason": status[n]["reason"], "focus": status[n]["focus"]} for n in rework}
        for n in rework:
            counts[n] = counts.get(n, 0) + 1
        why = "; ".join(f"{n}: {status[n]['reason']} → 재작업 {counts[n]}/{MAX_REWORK}" for n in rework)
        return rework, why, {"evidence_status": status, "rework_count": counts, "rework_request": requests}

    # 7. 판단
    short = [n for n, s in status.items() if not s["sufficient"]]
    why = (f"{', '.join(short)} 재작업 한도 도달, 정보 부족인 채로 판단" if short else "4개 관점 근거 충분") + " → judge"
    return ["judge"], why, {"evidence_status": status, "rework_request": {}}


def supervisor_node(state: dict) -> dict:
    step = state.get("step_count", 0) + 1
    nxt, reason, updates = decide(state, step)
    nodes = [] if nxt == END else list(nxt)
    log_decision(state.get("trace_id"), "supervisor", nodes or "END", reason, step=step,
                 company=(state.get("company") or {}).get("name"))
    return {"step_count": step, "next_nodes": nodes,
            "last_decision": f"[{step}] {', '.join(nodes) or 'END'} ← {reason}", **updates}


def route(state: dict):
    """add_conditional_edges("supervisor", route, ...)에 넣는다. 결정은 supervisor_node가 이미 State에 썼다."""
    return state.get("next_nodes") or END
