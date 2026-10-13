"""보고서 품질 평가 노드 (담당: 김다은)

보고서 생성 → 사실 검증 → 품질 평가 순서다. 사실 검증(verifier)이 수치·출처 번호를 원문과 대조한다면,
품질 평가는 보고서 전체가 아래 3개 항목을 지키는지 본다. 미달이면 supervisor가 reporter로 되돌린다.

평가 방식: Hybrid
- 코드 검사(결정적): 형식으로 판정할 수 있는 것 (인용 비율, 출처 번호, 과장·홍보·비하 표현, 발행처 수·편중, 같은 날 보도 편중)
- LLM Judge(내용): 코드로는 못 보는 것 (출처가 문장을 실제로 뒷받침하는지, 유리한 근거만 골랐는지 등).
  생성 모델(gpt-4.1-mini)과 다른 config.MODEL_JUDGE를 쓴다. 키가 없거나 호출이 실패하면 코드 검사만으로 판정한다
- 항목은 코드 검사와 Judge를 모두 통과해야 통과다

평가 항목
- groundedness  주장이 검색된 출처로 추적되는가
- neutrality    근거 없는 과장·추천·비하 없이 기준표와 점수로만 결정을 설명하는가 (투자/보류 결정 자체는 judge.decide 규칙)
- bias_control  단일 출처·한쪽 근거에 기대지 않는가 (확증편향 방지)

출력: {"eval_result": {"passed", "report_version", "mode", "items": {항목: {"passed", "reasons", ...}},
                       "fix_instructions", "judge_ran"}}
"""
import json
import logging
import os
import re
from typing import Literal

from pydantic import BaseModel, Field

from agents.reporter import REFERENCE_HEADING, cited_source_map, printed_reference_numbers
from agents.supervisor import evidence_report
from agents.verifier import _evidence_json, check_citations, find_wording_problems
from config import (LLM_ATTEMPTS, LLM_MAX_TOKENS, LLM_SEED, MAX_QUALITY_LOOPS, MODEL_JUDGE, QUALITY_CITED_RATIO,
                    QUALITY_MAX_SAME_DAY_SHARE, QUALITY_MAX_SOURCE_SHARE, QUALITY_MAX_UNVERIFIED,
                    QUALITY_MIN_PUBLISHERS)
from obs import log_event

logger = logging.getLogger(__name__)

USE_LLM_JUDGE = True
ITEMS = ("groundedness", "neutrality", "bias_control")
ITEM_LABELS = {"groundedness": "Groundedness", "neutrality": "중립성", "bias_control": "편향 통제"}

# 인용 비율을 따지는 분석 장 (reporter 목차의 장 제목에 들어 있는 말)
ANALYSIS_HEADINGS = ("기술 성숙도", "시장성", "이해관계자", "도메인 적용", "경쟁", "리스크")
RISK_HEADING = "리스크"
COMPETITION_HEADING = "경쟁 구도"

# 중립성: 기준표·점수 밖의 추천 문구와 경쟁사 비하 (과장 표현은 verifier.OVERCLAIM_WORDS를 같이 쓴다)
PROMOTIONAL = ("반드시 투자", "강력 추천", "강력히 추천", "투자해야 한다", "놓치지 말아야", "확실한 수익", "틀림없")
DISPARAGING = ("열등", "상대가 되지", "뒤처진 기업", "경쟁력이 없는")
# 수치 근거 없이 회사 보도자료에서 옮긴 홍보 표현
MARKETING = ("최적화된 성능", "최적의 성능", "혁신적인", "획기적인", "탁월한", "최고의", "최고 수준")
# 보고서에 나오면 안 되는 내부 자료 이름 (영문 필드명). reporter가 한글로 바꾸고, 남으면 미달
INTERNAL_NAMES = ("core_chips", "development_stage", "performance_metrics", "public_revenue_contracts",
                  "tech_summary", "market_analysis", "competitor_analysis",
                  "team_analysis", "summary_caution", "info_insufficient", "industry_context", "sub_market",
                  "segment_specific", "sources_for_citation", "evaluation_rules")
SUMMARY_EVIDENCE_LABEL = "**핵심 근거**"
SUMMARY_RISK_LABEL = "**핵심 리스크**"
# '정보 부족으로 점수가 제한되었다' 같은 서술. 정보 부족으로 채점된 항목이 없으면 채점 결과와 모순이다
INSUFFICIENT_CLAIM = re.compile(r"정보\s*부족[^.\n]{0,40}?(?:점수|채점|평가)[^.\n]{0,15}?(?:제한|낮아|낮게|하향)")

CITATION = re.compile(r"\[(\d+)\]")
REFERENCE_LINE = re.compile(r"^- \[(\d+)\] ([^(\n]+?)\(([^)]*)\)", re.MULTILINE)
DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
UNVERIFIED_MARK = "(확인 필요)"
NOTE_HEADING = "### 품질 평가 미달 항목"


# ---------- 보고서 읽기 ----------
def split_report(report: str) -> tuple[str, str]:
    body, _, reference = report.partition(REFERENCE_HEADING)
    return body, reference


def sections(body: str) -> list[tuple[str, str]]:
    """'## 제목' 단위로 나눈다. (제목, 본문) 목록."""
    out, title, lines = [], "", []
    for line in body.splitlines():
        if line.startswith("## "):
            if title:
                out.append((title, "\n".join(lines)))
            title, lines = line[3:].strip(), []
        else:
            lines.append(line)
    if title:
        out.append((title, "\n".join(lines)))
    return out


def paragraphs(text: str) -> list[str]:
    """표·소제목을 뺀 문단."""
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text)]
    return [b for b in blocks if b and not b.startswith(("|", "#", "{{"))]


def prose(text: str) -> str:
    return "\n".join(paragraphs(text))


def references(report: str) -> dict[int, dict]:
    """REFERENCE의 [n] → {"publisher", "kind", "date"}. kind는 바로 위 '### 기관 보고서'/'### 웹페이지'로 정한다."""
    _, reference = split_report(report)
    refs, kind = {}, ""
    for line in reference.splitlines():
        if line.startswith("### "):
            kind = "report" if "보고서" in line else "web"
        m = REFERENCE_LINE.match(line)
        if m:
            refs[int(m.group(1))] = {"publisher": m.group(2).strip(), "kind": kind, "date": m.group(3).strip()}
    return refs


def is_recommend(state: dict) -> bool:
    return (state.get("scores") or {}).get("decision") == "투자"


# ---------- 코드 검사 ----------
def check_groundedness(report: str, recommend: bool) -> list[str]:
    body, _ = split_report(report)
    reasons = [f"REFERENCE에 없는 출처 번호 {p['value']}" for p in check_citations(report)]
    if recommend:
        analysis = [t for h, t in sections(body) if any(k in h for k in ANALYSIS_HEADINGS)]
        numeric = [p for t in analysis for p in paragraphs(t) if re.search(r"\d", CITATION.sub("", p))]
        if numeric:
            cited = sum(1 for p in numeric if CITATION.search(p))
            if cited / len(numeric) < QUALITY_CITED_RATIO:
                reasons.append(f"분석 장에서 수치가 든 문단 {len(numeric)}개 중 인용이 있는 문단 {cited}개 "
                               f"({cited / len(numeric):.0%} < {QUALITY_CITED_RATIO:.0%})")
        # 경쟁 구도 표(점유율·비교 수치)는 코드가 만들어 문단 검사에 안 걸리므로 장 단위로 인용을 확인한다
        competition = [t for h, t in sections(body) if COMPETITION_HEADING in h]
        if competition and not CITATION.search(competition[0]):
            reasons.append("경쟁 구도 장에 인용 없음 (경쟁사 수치의 출처를 확인할 수 없음)")
    if recommend:
        # SUMMARY 핵심 근거·핵심 리스크: 줄마다 출처 번호 (분석 장 문단 검사에 안 걸리는 부분)
        summary = next((t for h, t in sections(body) if "SUMMARY" in h.upper()), "")
        for label in (SUMMARY_EVIDENCE_LABEL, SUMMARY_RISK_LABEL):
            if label not in summary:
                continue
            block = summary.split(label, 1)[1].split("**", 1)[0]
            bullets = [ln for ln in block.splitlines() if ln.strip().startswith(("-", "*", "•"))]
            uncited = [b for b in bullets if not CITATION.search(b)]
            if uncited:
                reasons.append(f"SUMMARY {label.strip('*')} {len(bullets)}개 중 {len(uncited)}개에 출처 번호 없음")
    exposed = [w for w in INTERNAL_NAMES if re.search(rf"(?<![A-Za-z_]){w}(?![A-Za-z_])", body)]
    exposed += [w for w in ("weaknesses", "strengths")   # 일반 단어이기도 해서 필드명처럼 쓰인 경우만
                if re.search(rf"(?<![A-Za-z_]){w}(?=['’\"]|\s*[가-힣])", body)]
    if exposed:
        reasons.append(f"내부 자료 이름 노출: {', '.join(exposed)}")
    marks = body.count(UNVERIFIED_MARK)
    if marks > QUALITY_MAX_UNVERIFIED:
        reasons.append(f"사실 검증에서 확인되지 않은 표현 {marks}곳 (허용 {QUALITY_MAX_UNVERIFIED})")
    return reasons


# 'HBF(High Bandwidth Fabric)'처럼 약어 뒤 괄호에 영문 풀이를 붙인 표기
ACRONYM = re.compile(r"\b([A-Z]{2,6})\s*\(\s*([A-Z][A-Za-z\-]+(?:\s+[A-Za-z\-]+){1,5})\s*\)")


def check_acronyms(report: str, state: dict) -> list[str]:
    """약어 풀이가 분석 자료·출처 원문에 없으면 미달. 왜: 자료에 없는 'HBF(High Bandwidth Fabric)'를 LLM이 지어 썼다."""
    body, _ = split_report(report)
    evidence = (_evidence_json(state) + " " + " ".join(str(s.get("snippet", "")) for s in state.get("sources") or [])).lower()
    return [f"자료에 없는 약어 풀이 '{m.group(0)}'" for m in ACRONYM.finditer(body)
            if " ".join(m.group(2).split()).lower() not in " ".join(evidence.split())]


def check_score_statements(report: str, state: dict) -> list[str]:
    """본문이 '정보 부족으로 점수가 제한되었다'고 쓰는데 정보 부족으로 채점된 항목이 없으면 채점 결과와 모순이다."""
    body, _ = split_report(report)
    items = (state.get("scores") or {}).get("items") or {}
    m = INSUFFICIENT_CLAIM.search(body)
    if m and items and not any(v.get("insufficient") for v in items.values() if isinstance(v, dict)):
        return [f"'{m.group(0)}'라고 썼지만 정보 부족으로 채점된 항목이 없음 (채점 결과와 다름)"]
    return []


def check_neutrality(report: str, state: dict) -> list[str]:
    body, _ = split_report(report)
    reasons = [f"자료에 없는 과장 표현 '{p['value']}'" for p in find_wording_problems(report, state)
               if p["kind"] == "overclaim"]
    reasons += [f"기준표 밖 추천 표현 '{w}'" for w in PROMOTIONAL if w in body]
    reasons += [f"경쟁사 비하 표현 '{w}'" for w in DISPARAGING if w in body]
    reasons += [f"수치 근거 없는 홍보 표현 '{w}'" for w in MARKETING if w in body]
    risk = [t for h, t in sections(body) if RISK_HEADING in h]
    if not risk or len(prose(risk[0])) < 100:
        reasons.append("리스크 장이 없거나 100자 미만 (리스크 축소)")
    return reasons


def check_bias(report: str, recommend: bool) -> list[str]:
    body, _ = split_report(report)
    refs = references(report)
    cites = [int(n) for n in CITATION.findall(body) if int(n) in refs]
    if not recommend:
        return []   # 추천 기업 없음 보고서는 보류 사유 정리라 출처 다양성을 따지지 않는다
    reasons = []
    publishers = {refs[n]["publisher"] for n in set(cites)}
    if len(publishers) < QUALITY_MIN_PUBLISHERS:
        reasons.append(f"인용된 출처의 발행처 {len(publishers)}곳 (최소 {QUALITY_MIN_PUBLISHERS})")
    if cites:
        top = max(set(cites), key=cites.count)
        share = cites.count(top) / len(cites)
        if share > QUALITY_MAX_SOURCE_SHARE:
            reasons.append(f"출처 [{top}]이 인용의 {share:.0%} (상한 {QUALITY_MAX_SOURCE_SHARE:.0%})")
    kinds = {refs[n]["kind"] for n in set(cites)}
    if kinds and not {"report", "web"} <= kinds:
        reasons.append("기관 보고서와 웹 출처 중 한쪽만 인용")
    risk = [t for h, t in sections(body) if RISK_HEADING in h]
    if risk and not CITATION.search(risk[0]):
        reasons.append("리스크 장에 인용 없음 (불리한 근거 누락 여부 확인 불가)")
    # 여러 매체가 같은 날 같은 발표를 옮긴 기사는 발행처가 달라도 독립 출처가 아니다 (보도자료 재전송)
    web_days = [DAY.search(refs[n]["date"]).group(0) for n in set(cites)
                if refs[n]["kind"] == "web" and DAY.search(refs[n]["date"])]
    if len(web_days) >= 3:
        day = max(set(web_days), key=web_days.count)
        same = web_days.count(day)
        if same >= 3 and same / len(web_days) > QUALITY_MAX_SAME_DAY_SHARE:
            reasons.append(f"인용한 웹 기사 {len(web_days)}건 중 {same}건이 같은 날({day}) 보도 "
                           f"(같은 발표의 재전송일 수 있어 독립 출처가 부족)")
    return reasons


def code_checks(report: str, state: dict) -> dict[str, list[str]]:
    recommend = is_recommend(state)
    return {
        "groundedness": (check_groundedness(report, recommend) + check_score_statements(report, state)
                         + check_acronyms(report, state)),
        "neutrality": check_neutrality(report, state),
        "bias_control": check_bias(report, recommend),
    }


# ---------- LLM Judge ----------
class JudgeItem(BaseModel):
    passed: bool
    reasons: list[str] = Field(description="미달이면 보고서의 어느 문장이 왜 문제인지 구체적으로. 통과면 빈 목록")
    quotes: list[str] = Field(default_factory=list,
                              description="reasons와 같은 순서로, 문제 문장을 보고서에서 글자 그대로 옮긴 것")


class EvidenceGap(BaseModel):
    perspective: Literal["tech_summary", "market", "competitor", "team"]
    reason: str = Field(description="보고서를 고쳐 써서는 해결되지 않는 이유 (자료 자체에 근거가 없거나 한쪽 출처뿐)")


class QualityJudgement(BaseModel):
    groundedness: JudgeItem = Field(description="본문 주장(수치·사실·결론)이 인용한 출처와 분석 자료로 뒷받침되고, 회사 발언을 제3자 평가처럼 쓰지 않았는가")
    neutrality: JudgeItem = Field(description="투자/보류 결정을 기준표 점수로만 설명하고, 리스크를 축소하거나 경쟁사를 깎아내리지 않는가")
    bias_control: JudgeItem = Field(description="분석 자료에 있는 약점·리스크·반대 근거를 빼고 유리한 근거만 쓰지 않았는가")
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list,
                                             description="미달 원인이 문장이 아니라 분석 자료의 근거 부족인 관점. 없으면 빈 목록")


JUDGE_PROMPT = """너는 투자 평가 보고서의 품질 검토자다. 아래 [분석 자료]만 근거로 [보고서]를 3개 항목별로 판정하라.
보고서 문장을 고치지 말고, 항목마다 통과 여부와 미달 사유만 적는다. 사유는 보고서의 해당 문장이나 장을 짚어 구체적으로 쓴다.
투자/보류 결정은 규칙으로 정해진 것이므로 결정의 옳고 그름은 판정하지 않는다. 결정을 설명하는 방식만 본다.
미달은 아래처럼 내용에 문제가 있을 때만 판정한다. 표현·용어 선택·문체·분량은 미달 사유가 아니다.
- groundedness: 보고서의 사실·수치가 [분석 자료]나 인용한 출처와 다르거나, 자료에 없는 사실을 단정함,
  또는 같은 사실(양산·테이프아웃 시점, 수치 등)을 장마다 다른 값·시점으로 씀,
  또는 문장에 붙은 출처 번호의 [인용 출처 원문]이 그 문장을 뒷받침하지 않음 (다른 출처를 잘못 붙임, 리스크 문장 포함),
  또는 회사(대표 인터뷰·보도자료)가 한 말을 '평가된다'·'인정받는다'처럼 제3자 평가나 확인된 사실로 씀
  (경쟁사 분석의 speaker가 대상 기업·경쟁사인 내용),
  또는 원문의 전망·가능성·계획·MOU를 단정하거나 완료로 씀 (원문보다 강한 결론)
- neutrality: 점수·기준표가 아닌 근거로 결정을 정당화하거나, 리스크를 축소하거나, 경쟁사를 깎아내리거나,
  수치 근거 없는 회사 홍보 문구(보도자료 어투)를 사실처럼 옮김
- bias_control: [분석 자료]에 있는 약점·리스크·반대 근거를 보고서에서 뺌, 또는 한 발표를 옮긴 기사들에만 기댐
자료에 없는 내용을 없다고 밝힌 것은 미달 사유가 아니다.
미달 원인이 보고서 문장이 아니라 [분석 자료] 자체에 근거가 없거나 한 발표·한 출처뿐인 것이면, 그 관점
(tech_summary, market, competitor, team)과 이유를 evidence_gaps에 적는다. 다시 써서 고칠 수 있는 표현 문제는 넣지 않는다.
{mode_note}

[분석 자료 - JSON]
{materials}

[인용 출처 원문] (보고서 REFERENCE 번호별 근거 원문 일부)
{cited}

[보고서]
{report}"""


def _judge(judge_llm):
    """False=끄기, 객체=그것 사용, None=자동(키가 있으면 실제 LLM, 없으면 건너뜀). verifier와 같은 방식."""
    if judge_llm is False or not USE_LLM_JUDGE:
        return None
    if judge_llm is not None:
        return judge_llm
    if not os.environ.get("OPENAI_API_KEY"):
        logger.info("[품질 평가] OPENAI_API_KEY가 없어 LLM Judge를 건너뜁니다.")
        return None
    from langchain_openai import ChatOpenAI  # 필요할 때만 import

    return ChatOpenAI(model=MODEL_JUDGE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS)


def cited_snippets(report: str, state: dict, per_source: int = 300, limit: int = 6000) -> str:
    """REFERENCE 번호별 근거 원문 일부. Judge가 문장에 붙은 출처가 맞는지 볼 수 있게 한다."""
    lines = []
    for n, raws in sorted(cited_source_map(report, state).items()):
        text = " / ".join(r.get("snippet", "") for r in raws if r.get("snippet"))
        if text:
            lines.append(f"[{n}] {text[:per_source]}")
    return "\n".join(lines)[:limit] or "(근거 원문 없음)"


def evidence_segments(state: dict) -> list[str]:
    """Judge 판정을 다시 확인할 때 찾아볼 자료 조각: 분석 자료의 문자열 값과 현재 기업 출처의 근거 원문."""
    segments = []

    def walk(value):
        if isinstance(value, str):
            segments.append(value)
        elif isinstance(value, dict):
            for v in value.values():
                walk(v)
        elif isinstance(value, list):
            for v in value:
                walk(v)
    walk(json.loads(_evidence_json(state)))
    name = (state.get("company") or {}).get("name")
    segments += [s.get("snippet", "") for s in state.get("sources") or [] if s.get("company") == name]
    return [s for s in segments if s]


NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
WORD = re.compile(r"[가-힣A-Za-z]{2,}")


def supported_by_evidence(quote: str, segments: list[str]) -> bool:
    """문장의 숫자가 모두 들어 있고, 그 조각의 단어 2개 이상이 문장에 나오는 자료 조각이 있으면 True.
    (문장 쪽 단어에는 조사가 붙어 '레귤러스는'처럼 되므로 자료 쪽 단어가 문장 안에 있는지로 본다)
    숫자가 없는 문장은 코드로 확인할 수 없어 False (Judge 판정을 그대로 둔다)."""
    quote = CITATION.sub("", quote)
    numbers = [n.replace(",", "") for n in NUMBER.findall(quote)]
    if not numbers:
        return False
    for seg in segments:
        compact = seg.replace(",", "")
        if all(n in compact for n in numbers) and sum(1 for w in set(WORD.findall(seg)) if w in quote) >= 2:
            return True
    return False


def recheck_groundedness(verdict: JudgeItem, state: dict) -> tuple[JudgeItem, int]:
    """Judge가 '자료에 없다'고 한 문장을 자료에서 다시 찾아, 실제로 있으면 그 사유를 뺀다.
    같은 문장을 회차마다 다르게 판정한 일이 있어 코드로 한 번 더 확인한다. (수정된 판정, 뺀 사유 수)"""
    if verdict.passed or not verdict.quotes:
        return verdict, 0
    segments = evidence_segments(state)
    keep = [(r, q) for i, r in enumerate(verdict.reasons)
            for q in [verdict.quotes[i] if i < len(verdict.quotes) else ""]
            if not supported_by_evidence(q, segments)]
    dropped = len(verdict.reasons) - len(keep)
    return JudgeItem(passed=not keep, reasons=[r for r, _ in keep], quotes=[q for _, q in keep]), dropped


def llm_checks(report: str, state: dict, judge_llm) -> QualityJudgement:
    mode_note = ("" if is_recommend(state) else
                 "추천 기업 없음 보고서다. 편향 통제는 보류 사유를 공정하게 정리했는지로만 본다.")
    prompt = JUDGE_PROMPT.format(mode_note=mode_note, materials=_evidence_json(state)[:15000],
                                 cited=cited_snippets(report, state), report=report)
    chain = judge_llm.with_structured_output(QualityJudgement).with_retry(stop_after_attempt=LLM_ATTEMPTS)
    return chain.invoke(prompt)


# ---------- 노드 ----------
def add_quality_note(report: str, items: dict) -> str:
    """재작성 한도까지 미달이면 미달 항목을 한계점 장 끝(REFERENCE 앞)에 남긴다. 이미 있으면 다시 넣지 않는다."""
    if NOTE_HEADING in report:
        return report
    lines = [f"- {ITEM_LABELS[k]}: {'; '.join(items[k]['reasons'][:2])}" for k in ITEMS if not items[k]["passed"]]
    note = (f"{NOTE_HEADING}\n보고서를 {MAX_QUALITY_LOOPS}회 다시 썼으나 아래 항목이 품질 기준에 미달했다.\n"
            + "\n".join(lines))
    head, sep, tail = report.partition(REFERENCE_HEADING)
    if not sep:
        return report.rstrip() + "\n\n" + note + "\n"
    return head.rstrip() + "\n\n" + note + "\n\n" + sep + tail


def evidence_gaps(code: dict, judgement, state: dict) -> dict[str, str]:
    """다시 써서는 해결되지 않는 근거 부족 관점 {관점: 이유}. supervisor가 이 관점을 다시 조사한다.
    코드: 편향 통제가 출처 수·같은 날 보도로 미달이면, 관점별 충분성 판정에서 같은 문제가 있는 관점.
    Judge: evidence_gaps로 짚은 관점.
    """
    gaps = {}
    if any(("발행처" in r or "같은 날" in r) for r in code["bias_control"]):
        for node, status in evidence_report(state).items():
            if "publisher_diversity" in status["focus"]:
                gaps[node] = status["reason"]
    for gap in (judgement.evidence_gaps if judgement else []):
        gaps.setdefault(gap.perspective, gap.reason)
    return gaps


def fix_instructions(items: dict) -> str:
    lines = [f"- [{ITEM_LABELS[k]}] {r}" for k in ITEMS for r in items[k]["reasons"] if not items[k]["passed"]]
    return "\n".join(lines)


def quality_eval_node(state: dict, judge_llm=None) -> dict:
    report = state.get("report") or ""
    code = code_checks(report, state)
    judge, judgement, judge_error = _judge(judge_llm), None, ""
    if judge is not None:
        try:
            judgement = llm_checks(report, state, judge)
        except Exception as e:  # noqa: BLE001 - Judge 실패는 코드 검사만으로 판정하고 기록한다
            judge_error = f"{type(e).__name__}: {e}"[:200]
            logger.warning("[품질 평가] LLM Judge 실패 → 코드 검사만으로 판정: %s", judge_error)

    items, rechecked = {}, 0
    for k in ITEMS:
        verdict = getattr(judgement, k) if judgement else None
        if k == "groundedness" and verdict is not None:
            verdict, rechecked = recheck_groundedness(verdict, state)
        reasons = list(code[k]) + ([f"(Judge) {r}" for r in verdict.reasons] if verdict and not verdict.passed else [])
        items[k] = {"passed": not code[k] and (verdict is None or verdict.passed), "reasons": reasons,
                    "code_passed": not code[k], "judge_passed": None if verdict is None else verdict.passed}
    passed = all(v["passed"] for v in items.values())
    gaps = evidence_gaps(code, judgement, state) if not passed and is_recommend(state) else {}
    # 회차마다 미달 사유를 결정 로그에 남긴다 (보고서에는 마지막 회차 사유만 남으므로)
    log_event(state.get("trace_id"), "quality", "quality_eval", report_version=state.get("report_version"),
              passed=passed, failed={k: v["reasons"] for k, v in items.items() if not v["passed"]},
              judge_ran=judgement is not None, evidence_gaps=gaps)
    logger.info("[품질 평가] v%s %s: %s", state.get("report_version"), "통과" if passed else "미달",
                {k: v["passed"] for k, v in items.items()})
    out = {"eval_result": {
        "passed": passed, "report_version": state.get("report_version"),
        "mode": "recommend" if is_recommend(state) else "no_recommend",
        "items": items, "fix_instructions": fix_instructions(items),
        "judge_ran": judgement is not None, "judge_error": judge_error, "evidence_gaps": gaps,
        "judge_rechecked": rechecked,   # Judge가 '자료에 없음'이라 했지만 자료에서 찾아 뺀 사유 수
    }}
    # 재작성 한도에 도달한 미달이면 supervisor가 종료하므로, 미달 항목을 보고서 한계점에 남긴다 (verifier의 '확인 필요'와 같은 방식)
    if not passed and state.get("quality_loops", 0) >= MAX_QUALITY_LOOPS:
        out["report"] = add_quality_note(report, items)
    return out
