"""보고서 수치·시점 정확도 규칙 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_accuracy_rules.py -q

- judge가 근거 문장에 쓴 환산값은 사실 검증 정답지가 아니다 (보고서가 옮기면 불일치로 잡힌다)
- 경쟁사 차별점에는 투자 정보가 남지 않고, 프롬프트에 오늘 날짜·시점 규칙이 들어간다
"""
from agents import competitor, judge, market, verifier

STATE = {
    "company": {"name": "A사"},
    "market_analysis": {"figures": [{"kind": "시장규모", "value": "3,400백만달러", "year": "2026"}]},
    "scores": {"items": {"market": {"score": 5, "evidence": "CXL 시장은 3,400백만달러(3.4억 달러) 전망", "sources": []}},
               "total": 73.0, "decision": "투자"},
    "sources": [],
}


def test_judge_conversion_is_not_in_truth_set():
    evidence = verifier.build_evidence(STATE)
    report = "## 2. 시장성\nCXL 메모리 시장은 3,400백만달러(3.4억 달러) 전망이다.\n## REFERENCE\n"
    mismatches, _ = verifier.find_quantity_mismatches(report, evidence)
    assert [m["surface"] for m in mismatches] == ["3.4억 달러"]   # 원자료 표기 3,400백만달러는 통과
    assert 5.0 in evidence["points"] and 73.0 in evidence["points"]  # 점수 값은 계속 정답지에 있다


def test_prompts_carry_unit_and_forecast_rules():
    assert "괄호로 환산값을 덧붙이지 않는다" in judge.PROMPT
    assert "'~ 전망'으로 쓰고" in market.PROMPT


def test_competitor_prompt_has_today_and_time_rules():
    prompt = competitor.PROMPT.format(name="A사", segment="인프라", product="MX1", today="2026-10-13", results="")
    assert "오늘은 2026-10-13다" in prompt and "당초" in prompt and "가장 최근인 기사" in prompt



def test_every_internal_name_checked_by_quality_has_a_report_label():
    from agents import quality, reporter
    assert set(quality.INTERNAL_NAMES) <= set(reporter.FIELD_LABELS)


def test_judge_evidence_in_basis_table_has_no_internal_names():
    from agents import quality, reporter
    scores = {"items": {"market": {"score": 5, "sources": [], "evidence":
              "HBM 시장 33% 성장 전망. 세부 분야 전체 수치(segment_specific=true, sub_market=false)로 인정. info_insufficient 아님"}}}
    table = reporter.build_score_basis_tables(scores, [])
    assert not [w for w in quality.INTERNAL_NAMES if w in table], table
    assert "세부 분야 전체 수치로 인정" in table and "정보 부족 아님" in table


# ---------- 항목 점수 표기 대조 (SUMMARY에 점수표와 다른 점수가 쓰인 적이 있다) ----------
ITEM_SCORES = {"items": {k: {"score": v, "evidence": "", "sources": []} for k, v in
                         {"tech": 4, "market": 3, "team": 4, "traction": 4, "competition": 4, "deal": 4}.items()},
               "total": 74.0, "decision": "투자"}
SCORE_REPORT = ("## SUMMARY\n**확인 필요**\n- 기술·제품 성숙도 2점(정보 부족으로 제한) 확인 필요\n"
                "## 6. 투자 판단 결과\n시장성은 3점이다. 보류 조건 ① 핵심 역량 2점 미만: 창업자·팀 4점, 기술·제품 성숙도 4점.\n"
                "경쟁 우위 5점이 되려면 학회 발표가 필요하다.\n| 경쟁 우위 | 4/5 | 3점 조건에 더해 특허 | 4점 조건에 더해 |\n"
                "## REFERENCE\n- [1] 기술·제품 성숙도 1점 (출처 제목)\n")


def test_only_wrong_item_score_claims_are_found():
    from agents import reporter
    found = reporter.score_claim_mismatches(SCORE_REPORT, ITEM_SCORES)
    assert [(m["surface"], m["actual"]) for m in found] == [("기술·제품 성숙도 2점", 4)]


def test_format_check_asks_to_fix_wrong_item_score():
    from agents import reporter
    violations = reporter.check_format(SCORE_REPORT, scores=ITEM_SCORES)
    assert any("기술·제품 성숙도 2점" in v and "4점" in v for v in violations)
    assert not any("점수표의 실제 점수" in v for v in reporter.check_format(SCORE_REPORT))   # scores 없으면 검사 안 함


def test_verifier_flags_wrong_item_score_only_for_recommend_report():
    state = {"company": {"name": "A사"}, "scores": ITEM_SCORES, "report": SCORE_REPORT, "sources": [],
             "report_version": 1}
    out = verifier.verifier_node(state, judge_llm=False, retriever=False)["verify_result"]
    assert "기술·제품 성숙도 2점" in [m["value"] for m in out["mismatches"]]
    hold = {**state, "scores": {**ITEM_SCORES, "decision": "보류"}}
    out = verifier.verifier_node(hold, judge_llm=False, retriever=False)["verify_result"]
    assert "기술·제품 성숙도 2점" not in [m["value"] for m in out["mismatches"]]


def test_summary_rule_has_no_copyable_example_score():
    from agents import reporter
    assert "기술·제품 성숙도 2점(정보 부족으로 제한)" not in reporter.SYSTEM_PROMPT
    assert "summary_caution에 적힌 값을 그대로" in reporter.SYSTEM_PROMPT


# ---------- 경쟁사 차별점은 평가 대상 기업 것만 ----------
def test_differentiation_keeps_only_target_company_facts():
    analysis = {"competitors": [{"name": "Qualcomm Technologies, Inc."}, {"name": "Groq"}, {"name": "엑시나 (XCENA)"}],
                "differentiation": [{"text": t, "kind": "성능", "basis": "사실", "speaker": "제3자"} for t in (
                    "Groq는 LPU로 초저지연 추론을 지원한다.", "Qualcomm AI200 칩은 대용량 메모리를 지원한다.",
                    "엑시나는 CXL 3.0 SoC를 개발한다.", "하이퍼엑셀은 LPDDR5X 대역폭을 90% 활용한다.",
                    "엔비디아 H100 대비 에너지 효율 1.33배", "Groq보다 전력 효율이 높다.", "자체 프레임워크 보유")]}
    kept = [d["text"] for d in competitor._keep_own_differentiation(analysis, "하이퍼엑셀")["differentiation"]]
    assert kept == ["하이퍼엑셀은 LPDDR5X 대역폭을 90% 활용한다.", "엔비디아 H100 대비 에너지 효율 1.33배",
                    "Groq보다 전력 효율이 높다.", "자체 프레임워크 보유"]
    # 예전 형식(문자열 목록) 결과도 같은 기준으로 거른다
    legacy = {"competitors": [{"name": "Groq"}], "differentiation": ["Groq는 LPU를 만든다.", "자체 SDK 보유"]}
    assert competitor._keep_own_differentiation(legacy, "하이퍼엑셀")["differentiation"] == ["자체 SDK 보유"]


def test_competitor_prompts_say_differentiation_is_target_only():
    prompt = competitor.PROMPT.format(name="하이퍼엑셀", segment="데이터센터", product="LPU", today="2026-10-13", results="")
    assert "'하이퍼엑셀' 자신의 사실만" in prompt and "오늘은 2026-10-13다" in prompt
    assert "경쟁사를 주어로 한 문장은 차별점에 넣지 않는다" in competitor.TIER_NOTE


# ---------- 품질 미달 메모에도 내부 자료 이름을 남기지 않는다 ----------
def test_quality_note_translates_internal_names():
    from agents import quality, reporter
    items = {"groundedness": {"passed": True, "reasons": []}, "neutrality": {"passed": True, "reasons": []},
             "bias_control": {"passed": False, "reasons": ["(Judge) tech_summary.weaknesses 필드가 비어 있고 "
                                                           "info_insufficient: true로 명시, missing_fields: [\"weaknesses\"]"]}}
    note = quality.add_quality_note("## 8. 한계점\n본문.\n\n" + reporter.REFERENCE_HEADING + "\n- [1] x", items)
    assert not [w for w in quality.INTERNAL_NAMES if w in note] and "missing_fields" not in note and ": true" not in note
    assert "기술 요약.약점 필드" in note


# ---------- 채점 항목 밖 점수('약점 0점')와 기준표 칸 표시 ----------
def test_format_check_flags_zero_score_on_non_item():
    from agents import reporter
    report = ("## SUMMARY\n**확인 필요**\n- 약점(weaknesses) 항목 점수 0점, 다음 조건 확인 필요\n"
              "## 6. 투자 판단 결과\n환산 점수 70.0점, 10점 만점 아님\n## REFERENCE\n")
    assert reporter.invalid_score_claims(report) == ["약점(weaknesses) 항목 점수 0점"]
    assert any("0점은 채점 결과에 없는 점수" in v for v in reporter.check_format(report, scores=ITEM_SCORES))
    assert not reporter.invalid_score_claims("환산 점수 70.0점이고 100점 만점이다.")


def test_unverified_mark_skips_rubric_table_cell():
    report = ("## 6. 투자 판단 결과\n| 시장성 | 5/5 | 수십억 달러 이상 규모, 연평균 성장률 20% 이상 | – |\n\n"
              "시장성 5점은 수십억 달러 규모와 성장률에 근거한다.\n## REFERENCE\n")
    marked = verifier.mark_unverified(report, [{"kind": "spelled", "value": "십억 달러", "surface": "십억 달러"}])
    assert "수십억 달러 이상 규모" in marked and "| 수십억 달러(확인 필요)" not in marked
    assert "수십억 달러(확인 필요) 규모와" in marked
