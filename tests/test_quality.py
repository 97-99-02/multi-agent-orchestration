"""보고서 품질 평가 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_quality.py -q"""
import pytest

from agents import quality as q

INVEST = {"scores": {"decision": "투자", "total": 78}, "company": {"name": "A사"}, "report_version": 2}
HOLD = {"scores": {"decision": "보류", "total": 60}, "company": {"name": "A사"}, "report_version": 1}

FILLER = "근거 자료에 따르면 해당 내용은 공개 기사와 기관 보고서에서 함께 확인된다. " * 5


def section(title, cite="[1][3]", number="2025년 3월 테이프아웃을 완료했다"):
    return f"## {title}\n{number}{cite}. {FILLER}{cite}\n"


def make_report(**override):
    parts = {
        "summary": "## SUMMARY\nA사는 78점으로 투자 결정.\n",
        "tech": section("1. 사업 아이디어·기술 성숙도"),
        "market": section("2. 시장성", "[2][4]", "시장은 2030년 120억 달러 규모"),
        "stake": section("3. 이해관계자 (팀 구성·투자사)", "[5]", "대표는 15년 경력"),
        "domain": section("4. 도메인 적용 (고객·PoC·공급)", "[6]", "2025년 PoC 2건"),
        "risk": section("6. 사업 리스크", "[2][6]", "경쟁사 3곳이 양산 중"),
        "reference": ("## REFERENCE\n### 기관 보고서\n- [1] KDB미래전략연구소(2024). AI 반도체 동향. https://kdb.co.kr\n"
                      "- [2] 한국전자통신연구원(2026). 메모리 동향. https://etri.re.kr\n### 웹페이지\n"
                      "- [3] 전자신문(2026-06-01). 기사. etnews.com, https://etnews.com/1\n"
                      "- [4] 매일경제(2026-06-01). 기사. mk.co.kr, https://mk.co.kr/1\n"
                      "- [5] 뉴스1(2026-09-04). 인터뷰. news1.kr, https://news1.kr/1\n"
                      "- [6] 더벨(2026-01-15). 기사. thebell.co.kr, https://thebell.co.kr/1\n"),
    }
    parts.update(override)
    return "\n".join(parts.values())


def node(report, state=INVEST, judge_llm=False):
    return q.quality_eval_node({**state, "report": report}, judge_llm=judge_llm)["eval_result"]


# ---------- 통과 기준 ----------
def test_good_report_passes_all_code_checks():
    result = node(make_report())
    assert result["passed"] is True
    assert result["report_version"] == 2 and result["mode"] == "recommend"
    assert result["judge_ran"] is False and result["fix_instructions"] == ""


def test_report_parsing():
    refs = q.references(make_report())
    assert refs[1] == {"publisher": "KDB미래전략연구소", "kind": "report"}
    assert refs[3] == {"publisher": "전자신문", "kind": "web"}


# ---------- Groundedness ----------
def test_uncited_numeric_paragraphs_fail_groundedness():
    report = make_report(tech=section("1. 사업 아이디어·기술 성숙도", cite=""),
                         market=section("2. 시장성", cite="", number="시장은 2030년 120억 달러"))
    result = node(report)
    assert not result["items"]["groundedness"]["passed"]
    assert "인용이 있는 문단" in result["items"]["groundedness"]["reasons"][0]


def test_citation_missing_from_reference_fails():
    result = node(make_report(stake=section("3. 이해관계자 (팀 구성·투자사)", "[9]")))
    assert any("[9]" in r for r in result["items"]["groundedness"]["reasons"])


def test_too_many_unverified_marks_fail():
    tech = section("1. 사업 아이디어·기술 성숙도") + "100 TOPS(확인 필요), 5W(확인 필요), 3nm(확인 필요)[1]\n"
    result = node(make_report(tech=tech))
    assert any("확인되지 않은 표현 3곳" in r for r in result["items"]["groundedness"]["reasons"])


# ---------- 중립성 ----------
def test_promotional_and_disparaging_words_fail_neutrality():
    result = node(make_report(summary="## SUMMARY\n반드시 투자해야 할 기업. 경쟁사는 열등하다.\n"))
    reasons = result["items"]["neutrality"]["reasons"]
    assert any("반드시 투자" in r for r in reasons) and any("열등" in r for r in reasons)


def test_overclaim_allowed_only_when_in_materials():
    summary = "## SUMMARY\n국내 최초 CXL 3.0 칩 개발.\n"
    assert not node(make_report(summary=summary))["items"]["neutrality"]["passed"]
    with_material = {**INVEST, "tech_summary": {"strengths": ["국내 최초 CXL 3.0 칩"]}}
    assert node(make_report(summary=summary), state=with_material)["items"]["neutrality"]["passed"]


def test_missing_risk_chapter_fails_neutrality():
    result = node(make_report(risk=""))
    assert any("리스크" in r for r in result["items"]["neutrality"]["reasons"])


# ---------- 편향 통제 ----------
def test_single_publisher_and_dominant_source_fail_bias():
    one = {k: section(t, "[3]") for k, t in [("tech", "1. 사업 아이디어·기술 성숙도"), ("market", "2. 시장성"),
                                            ("stake", "3. 이해관계자"), ("domain", "4. 도메인 적용"),
                                            ("risk", "6. 사업 리스크")]}
    reasons = node(make_report(**one))["items"]["bias_control"]["reasons"]
    assert any("발행처 1곳" in r for r in reasons)
    assert any("출처 [3]이 인용의 100%" in r for r in reasons)
    assert any("한쪽만" in r for r in reasons)


def test_risk_without_citation_fails_bias():
    result = node(make_report(risk=section("6. 사업 리스크", cite="", number="경쟁 심화 우려")))
    assert any("리스크 장에 인용 없음" in r for r in result["items"]["bias_control"]["reasons"])


# ---------- 관점 커버리지 ----------
def test_missing_or_thin_perspective_fails_coverage():
    report = make_report(domain="", stake="## 3. 이해관계자\n대표 15년 경력.\n")
    reasons = node(report)["items"]["coverage"]["reasons"]
    assert "'도메인 적용' 관점 장 없음" in reasons
    assert any("'이해관계자' 장" in r and "최소" in r for r in reasons)
    assert "'이해관계자' 장에 인용 없음" in reasons


def test_old_rag_toc_fails_coverage():
    # RAG 과제 목차(사업 아이디어 / 시장 규모 / 팀의 구성)는 관점 이름이 없어 커버리지 미달 → 9단계에서 목차 변경
    report = make_report(tech=section("1. 사업 아이디어 (핵심 컨셉)"), market=section("2. 시장 규모"),
                         stake=section("3. 팀의 구성"), domain="")
    assert len(node(report)["items"]["coverage"]["reasons"]) == 4


def test_no_recommend_report_skips_company_level_checks():
    report = ("## SUMMARY\n추천 기업 없음.\n## 1. 평가 개요\n후보 2곳 평가.\n"
              f"## 3. 사업 리스크\n{FILLER}\n## 4. 한계점\n정보 부족.\n")
    result = node(report, state=HOLD)
    assert result["mode"] == "no_recommend"
    assert result["items"]["coverage"]["passed"] and result["items"]["bias_control"]["passed"]
    assert result["passed"]


# ---------- LLM Judge (Hybrid) ----------
class FakeJudge:
    def __init__(self, fail=(), error=None):
        self.fail, self.error, self.prompts = set(fail), error, []

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **_):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return q.QualityJudgement(**{k: q.JudgeItem(passed=k not in self.fail,
                                                    reasons=["도메인 적용 장에 고객 실적 근거 없음"] if k in self.fail else [])
                                     for k in q.ITEMS})


def test_judge_failure_fails_item_even_if_code_passes():
    judge = FakeJudge(fail={"coverage"})
    result = node(make_report(), judge_llm=judge)
    item = result["items"]["coverage"]
    assert result["judge_ran"] and not result["passed"]
    assert item["code_passed"] is True and item["judge_passed"] is False
    assert item["reasons"] == ["(Judge) 도메인 적용 장에 고객 실적 근거 없음"]
    assert result["fix_instructions"] == "- [관점 커버리지] (Judge) 도메인 적용 장에 고객 실적 근거 없음"
    assert "[보고서]" in judge.prompts[0]


def test_code_failure_is_not_overridden_by_judge_pass():
    result = node(make_report(risk=""), judge_llm=FakeJudge())
    assert not result["items"]["neutrality"]["passed"]


def test_judge_error_falls_back_to_code_checks():
    result = node(make_report(), judge_llm=FakeJudge(error=RuntimeError("rate limit")))
    assert result["passed"] is True
    assert result["judge_ran"] is False and "rate limit" in result["judge_error"]


def test_no_api_key_skips_judge(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert q._judge(None) is None
