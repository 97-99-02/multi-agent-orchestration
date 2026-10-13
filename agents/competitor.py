"""경쟁사 비교 에이전트 (담당: 이채목)

입력: state["company"] (name, segment, product)
도구: 웹 검색 (Tavily)
LLM: config.MODEL_ANALYZE
출력: {"competitor_analysis": {...}, "sources": [출처 dict, ...]}
  - competitor_analysis: 같은 세부 분야 경쟁사 목록, 차별점, 경쟁 리스크, 직접 비교할 수 없는 항목
    경쟁사마다 tier를 붙인다: "선도 기업"(대기업·상장사, 시장 진입 위협) / "동급 기업"(비상장 스타트업, 상대적 위치 비교)
    비교·차별점은 주장 단위로 남긴다: 말한 주체(대상 기업·경쟁사·제3자), 근거 종류(수치·사실), 같은 조건 비교 여부, 근거 url
"""
import re
from datetime import date
from typing import Literal

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

import obs
from agents.explorer import SEGMENT_RULES
from agents.rework import previous, rework_of, union
from agents.web import format_results, search_many, source_date, to_source
from config import LLM_ATTEMPTS, LLM_MAX_TOKENS, LLM_SEED, MODEL_ANALYZE

# 세부 분야별 검색 키워드 (경쟁사를 같은 분야에서 찾기 위함)
SEGMENT_KEYWORDS = {
    "데이터센터": "데이터센터 AI 추론 가속기",
    "엣지": "온디바이스 엣지 AI NPU",
    "차량용": "차량용 자율주행 AI 반도체",
    "인프라": "CXL 메모리 인터커넥트 반도체",
    "기타": "AI 반도체",
}
# 해외 선도 기업은 국내 뉴스에 잘 잡히지 않아 영어 검색어를 함께 쓴다
SEGMENT_KEYWORDS_EN = {
    "데이터센터": "data center AI inference accelerator chip companies",
    "엣지": "edge AI NPU chip companies",
    "차량용": "automotive AI chip ADAS SoC companies",
    "인프라": "CXL switch memory interconnect chip companies",
    "기타": "AI chip companies",
}


Speaker = Literal["대상 기업", "경쟁사", "제3자"]
Basis = Literal["수치", "사실", "의견"]
StrengthKind = Literal["성능 수치", "전력 효율", "공정", "특허", "소프트웨어 개발 환경", "고객·양산", "제3자 검증"]


class Claim(BaseModel):
    text: str = Field(description="주장 한 문장. 검색 결과 하나에 있는 내용만")
    speaker: Speaker = Field(description="이 내용을 발표하거나 말한 주체. 회사 대표 인터뷰·보도자료를 옮긴 기사는 그 회사(대상 기업 또는 경쟁사), "
                                         "언론의 독자 분석·시장조사기관·벤치마크·고객 평가만 제3자")
    basis: Basis = Field(description="수치: 성능·전력·가격·공정·점유율 같은 숫자 / 사실: 출시·양산·고객·특허처럼 확인 가능한 사건 / "
                                     "의견: 위협적·경쟁 가능·우위·최초 가능성 같은 평가나 전망")
    comparable: bool = Field(description="대상 기업과 경쟁사 수치를 같은 측정 조건(같은 모델·정밀도·전력 조건)으로 비교했다고 원문에 나올 때만 true")
    evidence_id: int = Field(description="이 주장이 나온 검색 결과 번호 [n]")


class Strength(Claim):
    kind: StrengthKind = Field(description="차별점 종류")


class Competitor(BaseModel):
    name: str
    country: str = Field(description="국가. 예: 한국, 미국")
    product: str = Field(description="비교 대상 제품. 없으면 빈 문자열")
    summary: str = Field(description="그림 카드에 넣을 제품·사업 소개 한 구절 (한국어 40자 이내, 사실만, 평가·비교 표현 없이)")
    claims: list[Claim] = Field(description="대상 기업과 비교한 주장. 비교 근거가 없으면 빈 목록")


Tier = Literal["선도 기업", "동급 기업"]


class SegmentLabel(BaseModel):
    index: int = Field(description="기업 목록의 번호")
    in_segment: bool = Field(description="이 기업이 해당 세부 분야용 칩 제품을 판매하거나 개발 중인가 (검색 결과 근거)")
    tier: Tier = Field(description="선도 기업: 대기업(계열사 포함) 또는 코스피·코스닥·해외 증시 상장사 / 동급 기업: 비상장 스타트업")
    basis: str = Field(description="tier 근거 한 구절 (검색 결과 기준). 예: 코스닥 상장사, 삼성 계열 대기업, 비상장·시리즈B")


class SegmentLabels(BaseModel):
    labels: list[SegmentLabel]


LABEL_PROMPT = """아래 기업들이 '{segment}' 분야용 칩 제품을 판매하거나 개발 중인지, 그리고 기업 구분(tier)을 검색 결과를 근거로 판정하라.
{segment_rules}
여러 분야를 하는 대기업은 '{segment}' 분야 제품이 있으면 in_segment=true다. 근거가 없으면 false로 둔다.
장비(테스터·검사장비), 설계 IP 라이선스, 소재, 파운드리처럼 이 분야 기업에 납품하는 기업은 칩 제품을 파는 경쟁사가 아니므로 in_segment=false다.
기업 구분: 대기업(계열사 포함)이거나 코스피·코스닥·해외 증시에 상장한 기업이면 "선도 기업", 비상장 스타트업이면 "동급 기업".
스타트업으로 출발했더라도 이미 상장을 마쳤으면 "선도 기업"이다. 반대로 상장 추진, 상장 예비심사 청구, 프리IPO 투자,
IPO 준비 단계는 아직 비상장이므로 "동급 기업"이다. 주가·시가총액·상장일이 기사에 나오면 상장사로 본다.
기업 목록 (번호 그대로 index에 넣어라):
{names}

{results}"""


class CompetitorAnalysis(BaseModel):
    competitors: list[Competitor] = Field(description="같은 세부 분야의 국내외 경쟁사 3~6곳. 시장을 주도하는 대기업·상장사와 비슷한 단계의 스타트업을 모두 포함")
    differentiation: list[Strength] = Field(description="대상 기업의 기술·제품 차별점. 근거가 없으면 빈 목록")
    competitive_risks: list[str] = Field(description="경쟁 관점의 리스크. 경쟁사가 앞선 점(양산·고객·생태계 등) 포함")
    not_comparable: list[str] = Field(description="측정 조건이 달라 직접 비교할 수 없는 항목. '항목: 이유' 형식")
    evidence_ids: list[int] = Field(description="경쟁 리스크·비교 불가 항목의 근거로 사용한 검색 결과 번호 [n]")


PROMPT = """너는 AI 반도체 투자 심사역이다. '{name}'({segment}, 주력 제품: {product})의 경쟁 구도를 분석하라.
경쟁사는 같은 세부 분야({segment})에서 대상 제품과 같은 기능의 칩·모듈을 파는 기업으로 고른다. 다른 분야 기업과 비교하지 않는다.
장비(테스터·검사장비), 설계 IP 라이선스, 소재, 파운드리처럼 이 분야에 납품하는 기업과 고객사는 경쟁사가 아니다.
시장을 주도하는 대기업·상장사(진입 위협)와 비슷한 단계의 비상장 스타트업(상대적 위치)을 모두 찾아라.

비교(claims)와 차별점(differentiation)은 주장 단위로 쓴다. 주장 하나는 검색 결과 하나에 있는 내용만 쓰고 그 번호를 evidence_id에 넣는다.
- basis와 speaker를 원문 그대로 판정한다. 회사 대표·보도자료가 한 말은 제3자 평가가 아니다.
- '위협적', '충분히 경쟁 가능', '앞선 경쟁력', '최초가 될 가능성' 같은 평가는 basis=의견이다. 평가를 사실처럼 바꿔 쓰지 않는다.
- 수치가 있어도 측정 조건이 다르거나 알 수 없으면 comparable=false로 두고 우열을 쓰지 말고 각자의 수치만 쓴다.
  이런 항목은 not_comparable에 '항목: 이유'로 쓴다.
- 원문에 없는 연도·숫자를 넣지 않는다. 원문이 '올해'·'내년'이면 기사 게시일 연도 기준으로 쓴다.
- 비교할 근거가 없으면 claims를 비워 둔다. 차별점을 지어내지 않는다.
차별점은 성능 수치, 전력 효율, 공정, 특허, 소프트웨어 개발 환경(SDK, 컴파일러), 고객·양산, 제3자 검증만 쓴다.
투자 유치·기업가치·회사 소개는 차별점이 아니다.
경쟁 리스크에는 경쟁사가 앞선 점(양산, 고객, 생태계 등)도 쓴다.
회사 보도자료의 홍보 표현('최적화된 성능', '혁신적인', '획기적인', '탁월한')은 옮기지 않는다.
문장 안에 검색 결과 번호([1] 등)를 넣지 않는다. 번호는 evidence_id·evidence_ids에만 넣는다.
차별점(differentiation)은 '{name}' 자신의 사실만 '{name}'을 주어로 쓴다. 경쟁사의 소개나 강점은 차별점에 넣지 않는다
(경쟁사 내용은 그 경쟁사의 claims에 쓴다).

[시점 규칙] 오늘은 {today}다. 검색 결과마다 괄호 안 게시일을 기준으로 시점을 판단한다.
- 기사가 과거 발언이나 이전 계획을 인용한 것('당초 ~할 예정이었다', 작년 행사에서 밝힌 계획)은 현재 계획으로 쓰지 않는다.
- 이미 오늘 이전이 된 '예정'을 앞으로의 일처럼 쓰지 않는다.
- 같은 사실의 수치가 기사마다 다르면 게시일이 가장 최근인 기사를 따른다.

{results}"""


TIER_NOTE = """

[확정된 경쟁사와 기업 구분] (상장 여부 검색으로 확인한 값)
{tiers}
경쟁사는 위 기업만 이 순서대로 쓴다. 비교 주장과 경쟁 리스크에 기업 구분을 그대로 반영하라.
차별점(differentiation)은 평가 대상 기업 자신의 차별점만 쓴다. 경쟁사를 주어로 한 문장은 차별점에 넣지 않는다.
선도 기업은 대기업·상장사이므로 스타트업, 신생 기업, 비슷한 단계로 쓰지 않는다.
동급 기업은 비상장 스타트업이므로 대기업이나 상장사로 쓰지 않는다."""


def _rewrite_with_tier(prompt: str, analysis: dict, bases: list[str]) -> tuple[dict, list[int]]:
    """처음 분석은 tier 판정 전에 쓰여 구분과 어긋날 수 있다 (상장사 파두를 "유사 스타트업"으로 쓴 적이 있음).
    확정된 구분을 알려주고 같은 검색 결과로 비교 문장·차별점·리스크를 다시 쓴다."""
    comps = analysis["competitors"]
    tiers = "\n".join(f"- {x['name']}: {x['tier']} ({basis})" for x, basis in zip(comps, bases))
    a = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(CompetitorAnalysis).with_retry(stop_after_attempt=LLM_ATTEMPTS).invoke(
        prompt + TIER_NOTE.format(tiers=tiers))
    # 다시 쓸 때 기업 순서가 바뀐 적이 있어 번호가 아니라 이름으로 짝짓는다. 짝이 없는 기업은 처음 문장을 둔다
    rewritten = {_norm(new.name): new for new in a.competitors if _norm(new.name)}
    for x in comps:
        key = _norm(x["name"])
        partial = [t for k, t in rewritten.items() if k in key or key in k]
        # 정확히 같은 이름을 먼저 쓰고, 부분 일치는 후보가 하나일 때만 쓴다 ('삼성전자'와 '삼성전자 파운드리' 혼동 방지)
        match = rewritten.get(key) or (partial[0] if len(partial) == 1 else None)
        if match:
            x["summary"], x["claims"] = match.summary, [c.model_dump() for c in match.claims]
    analysis["differentiation"] = [d.model_dump() for d in a.differentiation]
    analysis["competitive_risks"] = a.competitive_risks
    analysis["not_comparable"] = a.not_comparable
    return analysis, a.evidence_ids


def _norm(s: str) -> str:
    return s.split("(")[0].replace(" ", "").lower()


NOT_FOUND = "같은 세부 분야 경쟁사를 공개 자료로 확인하지 못함"
def _name_key(name: str) -> str:
    """기업명 비교 키: 괄호 별칭을 빼고 첫 단어만, 소문자 ('Qualcomm Technologies, Inc.' → 'qualcomm', '엑시나 (XCENA)' → '엑시나')."""
    words = name.split("(")[0].split()
    return re.sub(r"[^0-9a-z가-힣]", "", words[0].lower()) if words else ""


def _keep_own_differentiation(analysis: dict, name: str) -> dict:
    """차별점에서 경쟁사를 주어로 한 문장('Groq는 …', '삼성전자는 …')을 뺀다. 경쟁사 이름으로 시작하더라도
    대상 기업 이름이 있거나 비교 표현('대비', '보다', '비해')이 있으면 대상 기업의 차별점으로 보고 남긴다
    (예: '엔비디아 H100 대비 전력 효율 1.33배'). 차별점 칸에 경쟁사 소개가 채워진 보고서가 연달아 나왔다."""
    target = _name_key(name)
    rivals = [k for k in (_name_key(c.get("name", "")) for c in analysis.get("competitors") or []) if len(k) >= 2]

    def own(item) -> bool:
        text = item.get("text", "") if isinstance(item, dict) else item   # 차별점은 주장 dict (예전 결과는 문자열)
        compact = re.sub(r"\s+", "", text).lower()
        if (target and target in compact) or re.search(r"대비|보다|비해", text):
            return True
        return not any(compact.startswith(k) for k in rivals)

    analysis["differentiation"] = [d for d in analysis.get("differentiation") or [] if own(d)]
    return analysis


# LLM이 문장 안에 남긴 검색 결과 번호 ([3], [2,5]). 보고서 출처 번호와 섞이면 다른 출처를 가리키게 된다
RESULT_REF = re.compile(r"\s*\[\d{1,3}(?:\s*[,~\-]\s*\d{1,3})*\]")


def _strip_refs(analysis: dict) -> dict:
    for c in analysis.get("competitors") or []:
        c["comparison"] = RESULT_REF.sub("", c.get("comparison", ""))
        for claim in c.get("claims") or []:
            claim["text"] = RESULT_REF.sub("", claim["text"])
    analysis["differentiation"] = [{**x, "text": RESULT_REF.sub("", x["text"])} if isinstance(x, dict) else RESULT_REF.sub("", x)
                                   for x in analysis.get("differentiation") or []]
    for key in ("competitive_risks", "not_comparable"):
        analysis[key] = [RESULT_REF.sub("", x) for x in analysis.get(key) or []]
    return analysis


# ── 주장 확인 ────────────────────────────────────────────────
# 왜: 대표 인터뷰의 "퀄컴이 가장 위협이 될 회사로 본다"가 "가장 위협적인 경쟁사로 평가된다"로 바뀌어 보고서에 실렸고,
# 사실 검증은 경쟁사 분석 결과를 정답지로 써서 통과시켰다. 기사 게시일보다 앞선 '2025년 4분기 양산 예정'도 들어갔다.
# 그래서 비교·차별점은 평가 의견을 버리고, 숫자·연도가 근거 원문에 있는 수치·사실만 남기며, 누가 발표했는지 표시한다.
NUMBER = re.compile(r"\d+(?:\.\d+)?")
FUNDING = re.compile(r"투자\s*(?:유치|금)|기업\s*가치|밸류|시리즈\s*[A-Z]|누적\s*투자")   # 차별점이 아닌 투자 정보
SPEAKER_LABEL = {"대상 기업": "{target} 발표", "경쟁사": "{owner} 발표", "제3자": ""}
NO_COMPARISON = "비교 근거 없음"


def _compact(text: str) -> str:
    return re.sub(r"[\s,]", "", text)


def facts_in_source(text: str, body: str, published: str = "") -> bool:
    """주장 속 숫자가 모두 근거 원문에 있는가. 연도는 원문의 '’26년'·'26년' 표기와, '올해'·'내년'을 게시 연도로 맞춰 본다."""
    compact = _compact(body)
    pub = int(published[:4]) if published[:4].isdigit() else None
    allowed = {str(pub)} if pub and "올해" in body else set()
    if pub and "내년" in body:
        allowed.add(str(pub + 1))
    for n in NUMBER.findall(_compact(text)):
        if n in compact or n in allowed:
            continue
        if len(n) == 4 and n.startswith("20") and (f"{n[2:]}년" in compact or any(f"{q}{n[2:]}" in compact for q in "'’‘")):
            continue
        return False
    return True


def check_claims(claims: list[dict], results: list[dict], strengths: bool = False) -> tuple[list[dict], list[str]]:
    """의견, 근거 번호가 틀린 주장, 숫자·연도가 원문에 없는 주장(차별점은 투자 정보도)을 버리고 근거 url·게시일을 붙인다.
    (남긴 주장, 버린 사유)"""
    kept, dropped = [], []
    for claim in claims:
        i = claim.pop("evidence_id", -1)
        reason = ("의견" if claim.get("basis") == "의견"
                  else "근거 번호 없음" if not 0 <= i < len(results)
                  else "투자 정보" if strengths and FUNDING.search(claim["text"])
                  else "")
        if not reason:
            r = results[i]
            published = source_date(r)
            if not facts_in_source(claim["text"], f"{r.get('title', '')} {r.get('content', '')}", published):
                reason = "숫자·연도가 원문에 없음"
        if reason:
            dropped.append(f"{reason}: {claim['text'][:60]}")
            continue
        kept.append({**claim, "url": r.get("url", ""), "date": published})
    return kept, dropped


def claim_text(claim: dict, target: str, owner: str = "") -> str:
    """보고서에 쓸 주장 문장. 회사가 발표한 내용이면 누가 발표했는지 붙인다."""
    label = SPEAKER_LABEL.get(claim.get("speaker"), "").format(target=target, owner=owner or "경쟁사")
    return f"{claim['text']} ({label})" if label else claim["text"]


def verify_claims(analysis: dict, results: list[dict], target: str) -> tuple[dict, list[int], list[str]]:
    """모든 비교·차별점 주장을 확인하고, 경쟁사마다 comparison(확인된 주장을 이은 문장)을 다시 만든다.
    (분석 결과, 남은 주장의 검색 결과 번호, 버린 사유)"""
    ids, dropped = [], []
    for c in analysis.get("competitors") or []:
        ids += [x.get("evidence_id", -1) for x in c.get("claims") or []]
        c["claims"], why = check_claims(c.get("claims") or [], results)
        dropped += [f"{c['name']} {w}" for w in why]
        c["comparison"] = " ".join(claim_text(x, target, c["name"]) for x in c["claims"]) or NO_COMPARISON
    ids += [x.get("evidence_id", -1) for x in analysis.get("differentiation") or []]
    analysis["differentiation"], why = check_claims(analysis.get("differentiation") or [], results, strengths=True)
    dropped += [f"차별점 {w}" for w in why]
    kept = ({x["url"] for c in analysis.get("competitors") or [] for x in c["claims"]}
            | {x["url"] for x in analysis["differentiation"]})
    used = [i for i in dict.fromkeys(ids) if 0 <= i < len(results) and results[i].get("url") in kept]
    return analysis, used, dropped


def rework_queries(name: str, product: str, segment: str, focus: list[str]) -> tuple[list[str], list[str]]:
    """Supervisor 재작업용 (general 검색어, news 검색어). 첫 실행은 국내 기사 3개 + 분야 영문 1개로 찾았으므로
    대상 제품 기준 영문 비교와 국내 업계 동향 기사로 찾는다."""
    subject = product or name
    general = [f"{subject} alternatives competitors comparison", f"{SEGMENT_KEYWORDS_EN.get(segment, 'AI chip companies')} startups funding"]
    news = [f"{SEGMENT_KEYWORDS.get(segment, 'AI 반도체')} 국내외 업체 경쟁 구도"]
    if "publisher_diversity" in focus:
        news.append(f"{name} 경쟁 제품 비교 평가")
    if "comparison_evidence" in focus:   # 확인된 비교 주장이 없을 때: 성능·전력 수치와 제3자 벤치마크를 찾는다
        general.append(f"{subject} TOPS power efficiency benchmark specifications")
        news.append(f"{name} {subject} 성능 전력 효율 수치")
    return general, news


def _merge_competitors(prev: list[dict], new: list[dict], target: str) -> list[dict]:
    """같은 경쟁사는 하나로 두고 주장을 합친다. (경쟁사 단위로 합치면 재작업에서 찾은 비교 주장이 버려졌다)"""
    merged: dict[str, dict] = {}
    for c in [*prev, *new]:
        key = _norm(c["name"])
        if key not in merged:
            merged[key] = {**c, **({"claims": list(c["claims"])} if "claims" in c else {})}
            continue
        kept = merged[key]
        if "claims" in c:
            kept["claims"] = union(kept.get("claims") or [], c["claims"], key=lambda x: _compact(x["text"]))
            kept["comparison"] = " ".join(claim_text(x, target, kept["name"]) for x in kept["claims"]) or NO_COMPARISON
        kept["summary"] = kept.get("summary") or c.get("summary", "")
    return list(merged.values())


def _merge_previous(prev: dict, new: dict, target: str = "") -> dict:
    """재작업 전에 확인한 경쟁사·차별점·리스크는 유지하고 새로 찾은 것만 더한다."""
    merged = dict(new)
    merged["competitors"] = _merge_competitors(prev.get("competitors") or [], new.get("competitors") or [], target)
    merged["differentiation"] = union(prev.get("differentiation") or [], new.get("differentiation") or [],
                                      key=lambda x: _compact(x["text"] if isinstance(x, dict) else x))
    merged["competitive_risks"] = union(prev.get("competitive_risks") or [], new.get("competitive_risks") or [])
    merged["not_comparable"] = union(prev.get("not_comparable") or [], new.get("not_comparable") or [])
    merged["excluded_other_segment"] = union(prev.get("excluded_other_segment") or [], new.get("excluded_other_segment") or [])
    if merged["competitors"]:
        merged["competitive_risks"] = [r for r in merged["competitive_risks"] if r != NOT_FOUND]
    return merged


def competitor_node(state: dict) -> dict:
    c = state["company"]
    name, segment, product = c["name"], c.get("segment", ""), c.get("product", "")
    kw = SEGMENT_KEYWORDS.get(segment, "AI 반도체")
    rework = rework_of(state, "competitor")
    if rework:
        general, news = rework_queries(name, product, segment, rework.get("focus") or [])
        results = search_many(news, max_results=5) + search_many(general, max_results=5, topic="general")
    else:
        results = search_many([
            f"{name} 경쟁사 비교",
            f"{kw} 시장 주요 업체",       # 선도 기업 (국내 기사)
            f"{kw} 스타트업 투자 유치",   # 동급 기업
        ], max_results=5)
        results += search_many([SEGMENT_KEYWORDS_EN.get(segment, "AI chip companies")], max_results=5, topic="general")
    prompt = PROMPT.format(name=name, segment=segment, product=product, today=date.today().isoformat(),
                           results=format_results(results))
    a = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(CompetitorAnalysis).with_retry(stop_after_attempt=LLM_ATTEMPTS).invoke(prompt)
    evidence_ids = list(a.evidence_ids)
    analysis = a.model_dump(exclude={"evidence_ids"})
    # 다른 세부 분야 기업은 비교 대상에서 뺀다 (설계 1.3: 경쟁사 비교는 세부 분야 기준).
    # 분야 판정은 대상 기업 정보를 주지 않은 별도 호출로 해, "같은 분야에서 고르라"는 지시에 맞춰
    # 분류가 기우는 것을 막는다.
    # 대상 기업 자신은 경쟁사에서 뺀다 ("파네시아 (Panmnesia)" 같은 표기도 걸러지도록 포함 관계로 비교)
    target = _norm(name)
    analysis["competitors"] = [x for x in analysis["competitors"]  # '퓨리오사'처럼 짧게 적힌 대상 기업도 뺀다
                               if not (len(_norm(x["name"])) >= 2 and (target in _norm(x["name"]) or _norm(x["name"]) in target))]
    names = [x["name"] for x in analysis["competitors"]]
    if names:
        listing = "\n".join(f"{i}. {n}" for i, n in enumerate(names))
        # 경쟁사마다 주력 칩과 상장 여부를 검색해 분야·기업 구분 판정의 근거로 쓴다
        # (칩 검색만으로는 상장 정보가 안 잡혀 코스닥 상장사 파두를 비상장으로 판정한 적이 있음)
        evidence = []
        for n in names:
            short = n.split("(")[0].strip()
            # 상장사는 주가·시가총액 기사가 있고 비상장사는 없다 ("상장 IPO"로 찾으면 상장 추진 기사가 섞였음)
            evidence += search_many([f"{short} {kw} 칩", f"{short} 주가 시가총액"], max_results=2, topic="general")
        labels = ChatOpenAI(model=MODEL_ANALYZE, temperature=0, seed=LLM_SEED, max_tokens=LLM_MAX_TOKENS).with_structured_output(SegmentLabels).with_retry(stop_after_attempt=LLM_ATTEMPTS).invoke(
            LABEL_PROMPT.format(segment=segment, segment_rules=SEGMENT_RULES, names=listing, results=format_results(evidence, limit=600)))
        label_of = {l.index: l for l in labels.labels}  # 이름 표기가 달라도 맞도록 번호로 매칭
        excluded = []
        for i, x in enumerate(analysis["competitors"]):
            label = label_of.get(i)
            x["in_segment"] = bool(label and label.in_segment)
            x["tier"] = label.tier if label else "선도 기업"
            x["basis"] = label.basis if label else ""
            if label and not label.in_segment:  # 판정이 빠진 기업은 '다른 분야'로 단정하지 않고 비교에서만 뺀다
                excluded.append(x["name"])
        analysis["excluded_other_segment"] = excluded
        analysis["competitors"] = [x for x in analysis["competitors"] if x.pop("in_segment")]
        bases = [x.pop("basis") for x in analysis["competitors"]]
        if analysis["competitors"]:
            analysis, more_ids = _rewrite_with_tier(prompt, analysis, bases)
            evidence_ids += more_ids
    analysis, claim_ids, dropped = verify_claims(analysis, results, name)
    if dropped:   # 버린 주장은 State 대신 결정 로그에 남긴다
        obs.log_diagnostics(state.get("trace_id"), "competitor", name, {"dropped_claims": dropped})
    if not analysis["competitors"]:
        analysis["competitive_risks"].append(NOT_FOUND)
    used = [results[i] for i in dict.fromkeys(evidence_ids + claim_ids) if 0 <= i < len(results)]
    analysis = _strip_refs(analysis)
    prev = previous(state, "competitor_analysis") if rework else None
    if prev:
        analysis = _merge_previous(prev, analysis, name)
    analysis = _keep_own_differentiation(analysis, name)   # 재작업 전 결과에서 넘어온 것까지 거른다
    return {
        "competitor_analysis": analysis,
        "sources": [to_source(r, name, "competitor") for r in used],
    }
