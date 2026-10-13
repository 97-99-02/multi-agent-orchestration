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
    assert refs[1] == {"publisher": "KDB미래전략연구소", "kind": "report", "date": "2024"}
    assert refs[3] == {"publisher": "전자신문", "kind": "web", "date": "2026-06-01"}


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


def test_no_recommend_report_skips_company_level_checks():
    report = ("## SUMMARY\n추천 기업 없음.\n## 1. 평가 개요\n후보 2곳 평가.\n"
              f"## 3. 사업 리스크\n{FILLER}\n## 4. 한계점\n정보 부족.\n")
    result = node(report, state=HOLD)
    assert result["mode"] == "no_recommend"
    assert result["items"]["bias_control"]["passed"]
    assert result["passed"]


# ---------- LLM Judge (Hybrid) ----------
class FakeJudge:
    def __init__(self, fail=(), error=None, gaps=(), quotes=()):
        self.fail, self.error, self.prompts, self.gaps, self.quotes = set(fail), error, [], list(gaps), list(quotes)

    def with_structured_output(self, schema):
        return self

    def with_retry(self, **_):
        return self

    def invoke(self, prompt):
        self.prompts.append(prompt)
        if self.error:
            raise self.error
        return q.QualityJudgement(**{k: q.JudgeItem(passed=k not in self.fail,
                                                    reasons=["자료의 양산 지연 리스크가 빠짐"] if k in self.fail else [],
                                                    quotes=self.quotes if k in self.fail else [])
                                     for k in q.ITEMS},
                                  evidence_gaps=[q.EvidenceGap(perspective=p, reason="공개 자료 없음") for p in self.gaps])


def test_judge_failure_fails_item_even_if_code_passes():
    judge = FakeJudge(fail={"bias_control"})
    result = node(make_report(), judge_llm=judge)
    item = result["items"]["bias_control"]
    assert result["judge_ran"] and not result["passed"]
    assert item["code_passed"] is True and item["judge_passed"] is False
    assert item["reasons"] == ["(Judge) 자료의 양산 지연 리스크가 빠짐"]
    assert result["fix_instructions"] == "- [편향 통제] (Judge) 자료의 양산 지연 리스크가 빠짐"
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


def test_items_are_three():
    assert q.ITEMS == ("groundedness", "neutrality", "bias_control")
    assert set(node(make_report())["items"]) == set(q.ITEMS)


# ---------- 다은님 피드백 반영 ----------
def test_competition_chapter_without_citation_fails_groundedness():
    competition = "## 5. 경쟁 구도\n| 구분 | 기업 |\n|---|---|\n| 선도 기업 | Qualcomm 15.6% |\n\n경쟁 의미 해설.\n"
    reasons = node(make_report(market=section("2. 시장성", "[2][4]", "시장 규모") + competition))["items"]["groundedness"]["reasons"]
    assert any("경쟁 구도 장에 인용 없음" in r for r in reasons)
    cited = competition + "경쟁사 정보 출처: [3][5]\n"
    reasons = node(make_report(market=section("2. 시장성", "[2][4]", "시장 규모") + cited))["items"]["groundedness"]["reasons"]
    assert not any("경쟁 구도" in r for r in reasons)


def test_marketing_phrase_fails_neutrality():
    result = node(make_report(summary="## SUMMARY\nNPU는 최적화된 성능을 제공한다.\n"))
    assert any("홍보 표현 '최적화된 성능'" in r for r in result["items"]["neutrality"]["reasons"])


def test_same_day_press_reprints_fail_bias():
    refs = ("## REFERENCE\n### 기관 보고서\n- [1] KDB(2024). 동향.\n- [2] ETRI(2026). 동향.\n### 웹페이지\n"
            "- [3] 전자신문(2026-08-19). 시리즈A. etnews.com\n- [4] 매일경제(2026-08-19). 시리즈A. mk.co.kr\n"
            "- [5] 뉴스1(2026-08-19). 시리즈A. news1.kr\n- [6] 더벨(2026-01-15). 기사. thebell.co.kr\n")
    reasons = node(make_report(reference=refs))["items"]["bias_control"]["reasons"]
    assert any("4건 중 3건이 같은 날(2026-08-19)" in r for r in reasons)


# ---------- 2차 피드백 (A단계) ----------
def test_summary_evidence_bullets_need_citations():
    summary = ("## SUMMARY\n**결론:** 투자.\n**핵심 근거**\n- 80 TOPS 칩 개발[1]\n- 시장 연 16% 성장\n- 팀 10년 경력\n"
               "**핵심 리스크**\n- 경쟁[2]\n")
    reasons = node(make_report(summary=summary))["items"]["groundedness"]["reasons"]
    assert any("SUMMARY 핵심 근거 3개 중 2개에 출처 번호 없음" in r for r in reasons)


def test_internal_field_names_fail_groundedness():
    risk = section("6. 사업 리스크", "[2][6]", "'weaknesses' 항목 정보가 없어 summary_caution 확인")
    reasons = node(make_report(risk=risk))["items"]["groundedness"]["reasons"]
    assert any("내부 자료 이름 노출: summary_caution, weaknesses" in r for r in reasons)
    english = section("6. 사업 리스크", "[2][6]", "Hailo competitive strengths in edge NPUs")
    assert not any("내부 자료" in r for r in node(make_report(risk=english))["items"]["groundedness"]["reasons"])


def test_best_level_phrase_is_marketing():
    result = node(make_report(summary="## SUMMARY\n국내 최고 수준의 기술력.\n"))
    assert any("'최고 수준'" in r for r in result["items"]["neutrality"]["reasons"])



# ---------- 근거 부족 관점 (B단계) ----------
def test_judge_evidence_gaps_are_returned_only_when_failed():
    result = node(make_report(), judge_llm=FakeJudge(fail={"bias_control"}, gaps=["market"]))
    assert result["evidence_gaps"] == {"market": "공개 자료 없음"}
    passed = node(make_report(), judge_llm=FakeJudge(gaps=["market"]))
    assert passed["passed"] and passed["evidence_gaps"] == {}


def test_same_day_bias_maps_to_perspective_with_same_day_sources():
    refs = ("## REFERENCE\n### 기관 보고서\n- [1] KDB(2024). 동향.\n- [2] ETRI(2026). 동향.\n### 웹페이지\n"
            "- [3] 전자신문(2026-08-19). 시리즈A. etnews.com\n- [4] 매일경제(2026-08-19). 시리즈A. mk.co.kr\n"
            "- [5] 뉴스1(2026-08-19). 시리즈A. news1.kr\n- [6] 더벨(2026-01-15). 기사. thebell.co.kr\n")
    same_day = [{"company": "A사", "node": "team", "kind": "web", "publisher": p, "date": "2026-08-19"}
                for p in ("etnews", "mk")]
    state = {**INVEST, "sources": same_day, "team_analysis": {"members": [{"name": "홍길동"}], "concerns": []}}
    result = node(make_report(reference=refs), state=state)
    assert "team" in result["evidence_gaps"] and "같은 날" in result["evidence_gaps"]["team"]



# ---------- Judge 판정 재확인, 인용 출처 원문 (C단계) ----------
SUPPORTED = {**INVEST, "tech_summary": {"development_stage": ["레귤러스 2026년 상반기 테이프아웃 계획"]}}


def test_quote_with_numbers_and_words_in_materials_is_supported():
    segments = q.evidence_segments(SUPPORTED)
    assert q.supported_by_evidence("레귤러스는 2026년 상반기 테이프아웃을 목표로 한다[3].", segments)
    assert not q.supported_by_evidence("레귤러스는 2024년 하반기 양산을 앞두고 있다.", segments)   # 숫자가 다름
    assert not q.supported_by_evidence("레귤러스 테이프아웃 상반기", segments)                    # 숫자 없으면 확인 불가


def test_judge_unsupported_claim_found_in_materials_is_dropped():
    judge = FakeJudge(fail={"groundedness"}, quotes=["레귤러스는 2026년 상반기 테이프아웃을 목표로 한다"])
    result = node(make_report(), state=SUPPORTED, judge_llm=judge)
    assert result["items"]["groundedness"]["passed"] and result["judge_rechecked"] == 1
    really_missing = FakeJudge(fail={"groundedness"}, quotes=["2024년 하반기 양산을 앞두고"])
    result = node(make_report(), state=SUPPORTED, judge_llm=really_missing)
    assert not result["items"]["groundedness"]["passed"] and result["judge_rechecked"] == 0


def test_judge_prompt_includes_cited_source_snippets():
    from agents import reporter
    src = {"company": "A사", "node": "team", "kind": "web", "title": "인터뷰", "publisher": "뉴스1",
           "date": "2026-09-04", "url": "https://news1.kr/1", "snippet": "대표는 해외 대비 3~4년 지연이라고 말했다"}
    state = {**INVEST, "sources": [src]}
    text = reporter.collect_sources(state)[0]["text"]
    report = make_report(reference=f"## REFERENCE\n### 웹페이지\n- [1] {text}\n")
    judge = FakeJudge()
    node(report, state=state, judge_llm=judge)
    assert "[1] 대표는 해외 대비 3~4년 지연" in judge.prompts[0]


# ---------- 3차 피드백 (F4) ----------
def test_summary_risk_bullets_need_citations():
    summary = ("## SUMMARY\n**결론:** 투자.\n**핵심 근거**\n- 80 TOPS 칩[1]\n- 시장 성장[2]\n- 팀 경력[5]\n"
               "**핵심 리스크**\n- 양산 지연 가능성[3]\n- 대기업 진입\n- 자금 소진 속도\n**확인 필요**\n- 없음\n")
    reasons = node(make_report(summary=summary))["items"]["groundedness"]["reasons"]
    assert any("SUMMARY 핵심 리스크 3개 중 2개에 출처 번호 없음" in r for r in reasons)
    assert not any("핵심 근거" in r for r in reasons)


def test_insufficient_score_statement_must_match_scores():
    # 지수 실행: 한계점은 '정보 부족으로 점수가 제한되었다'인데 정보 부족으로 채점된 항목은 없었다
    limits = (section("6. 사업 리스크", "[2][6]", "경쟁사 3곳이 양산 중")      # REFERENCE 앞에 오도록 리스크 장 뒤에 붙인다
              + section("8. 한계점", "[1]", "팀 규모는 공개되지 않아 정보 부족으로 점수가 제한되었다"))
    no_insufficient = {**INVEST, "scores": {**INVEST["scores"], "items": {"team": {"score": 4, "insufficient": False}}}}
    reasons = node(make_report(risk=limits), no_insufficient)["items"]["groundedness"]["reasons"]
    assert any("정보 부족으로 채점된 항목이 없음" in r for r in reasons)
    with_insufficient = {**INVEST, "scores": {**INVEST["scores"], "items": {"team": {"score": 2, "insufficient": True}}}}
    reasons = node(make_report(risk=limits), with_insufficient)["items"]["groundedness"]["reasons"]
    assert not any("정보 부족으로 채점된" in r for r in reasons)


def test_judge_checks_company_statements_and_stronger_conclusions():
    assert "회사(대표 인터뷰·보도자료)가 한 말을 '평가된다'" in q.JUDGE_PROMPT
    assert "원문보다 강한 결론" in q.JUDGE_PROMPT and "리스크 문장 포함" in q.JUDGE_PROMPT
