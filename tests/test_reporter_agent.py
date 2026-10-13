"""Agent 과제용 보고서 변경 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_reporter_agent.py -q

- 목차 1~4장 제목에 품질 평가의 4개 관점 이름이 들어가고, 바뀐 장 번호로 표가 들어간다
- 품질 평가 미달로 다시 쓸 때 수정 지시가 프롬프트에 붙는다
- 재작성 한도까지 미달이면 미달 항목이 한계점에 남는다
"""
from types import SimpleNamespace

from agents import quality, reporter
from config import MAX_QUALITY_LOOPS


class CaptureLLM:
    def __init__(self):
        self.prompts = []

    def invoke(self, messages):
        self.prompts.append(messages[-1][1])
        return SimpleNamespace(content="## SUMMARY\n추천 기업 없음.\n## 3. 사업 리스크\n리스크.\n## 4. 한계점\n정보 부족.\n")


HOLD = {"candidates": ["A사"], "company": {"name": "A사"}, "sources": [],
        "scores": {"decision": "보류", "total": 60, "hold_reasons": ["70점 미만"], "items": {}},
        "rejected": [{"company": "A사", "reason": "70점 미만", "total": 60}]}


def toc_titles():
    return [line[3:] for line in reporter.TOC_RECOMMEND.splitlines() if line.startswith("## ")]


# ---------- 목차 ----------
def test_toc_analysis_chapters_are_checked_by_quality():
    # 품질 평가가 인용 비율을 보는 분석 장 이름이 목차에 실제로 있어야 검사가 걸린다
    titles = " ".join(toc_titles())
    for heading in quality.ANALYSIS_HEADINGS:
        assert heading in titles, heading


def test_placeholders_sit_in_the_chapters_the_insert_patterns_look_for():
    titles = toc_titles()
    competitor = next(t for t in titles if reporter.COMPETITOR_PLACEHOLDER in t)
    score = next(t for t in titles if reporter.PLACEHOLDER in t)
    assert reporter.COMPETITOR_SECTION_PATTERN.match(f"## {competitor}\n")
    assert reporter.SCORE_SECTION_PATTERN.match(f"## {score}\n")


def test_tables_fall_back_to_new_chapter_numbers_when_placeholder_missing():
    body = "## 5. 경쟁 구도\n해설.\n## 6. 투자 판단 결과\n해설.\n"
    out = reporter._insert_table(body, reporter.COMPETITOR_PLACEHOLDER, "|표|", reporter.COMPETITOR_SECTION_PATTERN, "경쟁사 표")
    assert out.startswith("## 5. 경쟁 구도\n\n|표|")
    scores = {"items": {}, "total": 78, "decision": "투자"}
    out = reporter.insert_score_table(body, scores)
    assert "## 6. 투자 판단 결과\n\n" + reporter.build_score_table(scores) in out


def test_risk_chapter_rule_asks_for_citations():
    assert "'사업 리스크' 장의 리스크마다 근거 출처 번호" in reporter.SYSTEM_PROMPT


# ---------- 품질 수정 지시 ----------
def run(state):
    llm = CaptureLLM()
    reporter.reporter_node(state, llm=llm)
    return llm.prompts[0]


def test_quality_fix_instructions_are_added_for_previous_version():
    state = {**HOLD, "report_version": 2,
             "eval_result": {"passed": False, "report_version": 1, "fix_instructions": "- [편향 통제] 리스크 장에 인용 없음"}}
    prompt = run(state)
    assert "[품질 수정 요청]" in prompt and "리스크 장에 인용 없음" in prompt


def test_no_quality_notes_when_passed_or_stale():
    passed = {**HOLD, "report_version": 2, "eval_result": {"passed": True, "report_version": 1, "fix_instructions": ""}}
    stale = {**HOLD, "report_version": 3,
             "eval_result": {"passed": False, "report_version": 1, "fix_instructions": "- [중립성] x"}}
    assert "[품질 수정 요청]" not in run(passed)
    assert "[품질 수정 요청]" not in run(stale)


def test_halted_reason_is_passed_to_materials():
    materials = reporter.build_materials({**HOLD, "last_error": "supervisor 실행 40회 초과"}, [])
    assert materials["halted"] == "supervisor 실행 40회 초과"
    assert "halted" not in reporter.build_materials(HOLD, [])


# ---------- 분량·파일명 ----------
def test_page_limit_is_ten_pages():
    head = "## SUMMARY\n요약\n## 1. 본문\n"
    tail = "\n## REFERENCE\n- [1] 출처(2026). 제목."
    ok = head + "가" * (reporter.PAGE_CHARS * 9) + tail
    over = head + "가" * (reporter.PAGE_CHARS * 10) + tail
    assert not any("장 기준" in v for v in reporter.check_format(ok))
    assert any("10장 기준" in v for v in reporter.check_format(over))


def test_output_filename_is_for_agent_assignment():
    name = reporter.build_output_filename()
    assert name.startswith("Agent-Output_") and name.endswith(".pdf")


# ---------- 재작성 한도까지 미달 ----------
REPORT = "## SUMMARY\n요약\n## 8. 한계점\n정보 부족.\n## REFERENCE\n- [1] 출처(2026). 제목."
INVEST = {"scores": {"decision": "투자"}, "company": {"name": "A사"}, "report": REPORT, "report_version": 3}


def test_quality_note_is_added_when_loops_are_exhausted():
    out = quality.quality_eval_node({**INVEST, "quality_loops": MAX_QUALITY_LOOPS}, judge_llm=False)
    assert not out["eval_result"]["passed"]
    report = out["report"]
    assert report.index(quality.NOTE_HEADING) < report.index("## REFERENCE")
    assert "중립성" in report
    assert quality.add_quality_note(report, out["eval_result"]["items"]) == report   # 두 번 넣지 않음


def test_no_quality_note_before_loop_limit():
    out = quality.quality_eval_node({**INVEST, "quality_loops": MAX_QUALITY_LOOPS - 1}, judge_llm=False)
    assert "report" not in out


# ---------- 11단계 1회차 실행에서 찾은 것 ----------
def test_grouped_citations_are_split_before_renumbering():
    assert reporter.split_grouped_citations("장벽이 높다[6,7].") == "장벽이 높다[6][7]."
    assert reporter.split_grouped_citations("경쟁[8, 9,10,11]") == "경쟁[8][9][10][11]"
    assert reporter.split_grouped_citations("성장[3-5]") == "성장[3][4][5]"
    assert reporter.split_grouped_citations("기간[2024-2030] 전망[2]") == "기간[2024-2030] 전망[2]"   # 연도는 그대로


def test_split_citations_follow_renumbered_reference():
    sources = [{"n": n, "label": "웹페이지", "title": f"기사{n}"} for n in range(1, 12)]
    body = reporter.split_grouped_citations("A[6,7]. B[8,9,10,11].")
    body, cited = reporter.keep_cited_sources(body, sources)
    assert body == "A[1][2]. B[3][4][5][6]."                       # 본문 번호와 REFERENCE 번호가 같이 바뀐다
    assert [s["title"] for s in cited] == ["기사6", "기사7", "기사8", "기사9", "기사10", "기사11"]


# ---------- PDF 분량 (최소 7장 ~ 최대 10장) ----------
SHORT = ("## SUMMARY\n**결론:** 요약.\n## 1. 사업 아이디어·기술 성숙도\n" + "기술 내용 문장이다[1]. " * 40
         + "\n## 7. 사업 리스크\n리스크[1].\n## REFERENCE\n### 웹페이지\n- [1] 매체(2026). 기사. x.com, https://x.com\n")


def test_short_report_is_typeset_with_larger_spacing(tmp_path, monkeypatch):
    _, scaled = reporter.export_report_pdf(SHORT, output_dir=tmp_path, filename="scaled.pdf")
    monkeypatch.setattr(reporter, "LAYOUT_SCALES", (1.0,))
    _, base = reporter.export_report_pdf(SHORT, output_dir=tmp_path, filename="base.pdf")
    assert scaled >= base
    assert reporter._LAYOUT_SCALE == 1.0          # 조판이 끝나면 배율을 되돌린다


def test_body_starts_on_new_page_only_when_first_page_is_full(tmp_path):
    ref = "## REFERENCE\n### 웹페이지\n- [1] 매체(2026). 기사.\n"
    body = "## 1. 사업 아이디어·기술 성숙도\n짧은 본문[1].\n"
    full_summary = "## SUMMARY\n" + "".join(f"- 근거 문장 {i}번은 자료에 있는 사실이다.\n" for i in range(22))
    _, pages = reporter.export_report_pdf(full_summary + body + ref, output_dir=tmp_path, filename="full.pdf")
    _, short_pages = reporter.export_report_pdf("## SUMMARY\n요약.\n" + body + ref, output_dir=tmp_path,
                                                filename="short.pdf")
    assert pages > short_pages        # 1쪽이 SUMMARY로 차면 본문은 새 쪽, 짧으면 이어서 (빈 쪽을 만들지 않음)


def test_agent_result_numbers_are_removed_from_table_and_materials():
    # 경쟁사 에이전트가 문장에 남긴 자기 검색 결과 번호가 보고서 출처 번호로 오인되지 않게 지운다
    state = {"competitor_analysis": {"competitors": [{"name": "Qualcomm", "tier": "선도 기업", "country": "미국",
                                                      "product": "Hexagon", "comparison": "점유율 15.6%[3]"}],
                                     "differentiation": ["2026년 하반기 양산 계획 [19]"], "competitive_risks": []},
             "company": {"name": "A사"}, "scores": {"decision": "투자"}}
    table = reporter.build_competitor_table(state)
    assert "점유율 15.6%" in table and "[3]" not in table and "[19]" not in table
    materials = reporter.build_materials({**state, **{k: {} for k in ("tech_summary", "market_analysis", "team_analysis")},
                                          "candidates": ["A사"]}, [])
    assert materials["competitor_analysis"]["competitors"][0]["comparison"] == "점유율 15.6%"


def test_competitor_table_has_no_bare_source_line_and_sources_show_their_agent():
    state = {"competitor_analysis": {"competitors": [], "differentiation": [], "competitive_risks": []}}
    body = reporter.insert_competitor_table("## 5. 경쟁 구도\n{{COMPETITOR_TABLE}}\n해설[2].\n", state)
    assert "경쟁사 정보 출처" not in body
    sources = [{"n": 2, "label": "웹페이지", "title": "경쟁 기사", "raw": [{"node": "competitor"}, {"node": "team"}]}]
    full = {**state, "company": {"name": "A사"}, "scores": {"decision": "투자"}, "candidates": ["A사"],
            **{k: {} for k in ("tech_summary", "market_analysis", "team_analysis")}}
    assert reporter.build_materials(full, sources)["sources_for_citation"][0]["used_by"] == ["competitor", "team"]


def test_key_name_citations_are_removed():
    assert reporter.KEY_CITATION_PATTERN.sub("", "정보는 확인되지 않는다[team_analysis].") == "정보는 확인되지 않는다."
    assert reporter.KEY_CITATION_PATTERN.sub("", "근거[3].") == "근거[3]."


def test_prompt_rules_for_rounded_amounts_and_marketing_words():
    assert "'수십억 달러'" in reporter.SYSTEM_PROMPT and "어림 표현" in reporter.SYSTEM_PROMPT
    assert "홍보 표현" in reporter.SYSTEM_PROMPT
    assert "대괄호 안에는 숫자만" in reporter.SYSTEM_PROMPT
    assert "used_by에 competitor" in reporter.SYSTEM_PROMPT



def test_competitor_agent_strips_result_numbers():
    from agents import competitor
    analysis = {"competitors": [{"name": "Hailo", "comparison": "TOPS 우위[2, 5]"}],
                "differentiation": ["특허 12건 [4]"], "competitive_risks": ["가격 경쟁[1]"]}
    out = competitor._strip_refs(analysis)
    assert out["competitors"][0]["comparison"] == "TOPS 우위"
    assert out["differentiation"] == ["특허 12건"] and out["competitive_risks"] == ["가격 경쟁"]



def test_key_name_only_bullets_are_removed():
    body = "**확인 필요**\n- 팀 역량 확인 필요\n- summary_caution 없음\n## 1. 장\n"
    assert reporter.KEY_NAME_LINE_PATTERN.sub("", body) == "**확인 필요**\n- 팀 역량 확인 필요\n## 1. 장\n"


def test_summary_is_not_scaled_and_body_follows_dashboard_overflow(tmp_path, monkeypatch):
    # 간격 배율이 커도 SUMMARY 스타일은 배율 1, 다른 스타일은 배율 적용
    monkeypatch.setattr(reporter, "_LAYOUT_SCALE", 1.45)
    font, bold = reporter.register_korean_font()
    assert reporter._styles(font, bold, scaled=False)["body"].leading < reporter._styles(font, bold)["body"].leading
    assert reporter._LAYOUT_SCALE == 1.45


def test_field_names_are_translated_but_english_words_kept():
    body = "'weaknesses' 항목과 public_revenue_contracts 정보, strengths가 없다. Hailo competitive strengths."
    assert reporter.translate_field_names(body) == "'약점' 항목과 공개 매출·계약 정보, 강점가 없다. Hailo competitive strengths."


def test_prompt_rules_for_consistency_and_summary_citations():
    assert "모든 장에서 같은 값·시점" in reporter.SYSTEM_PROMPT
    assert "줄마다 끝에 출처 번호" in reporter.SYSTEM_PROMPT


# ---------- 지적된 장만 재작성 (2차 피드백 C단계) ----------
PREV_REPORT = """## SUMMARY
**결론:** 투자.

## 1. 사업 아이디어·기술 성숙도
기술 내용[2].

## 5. 경쟁 구도
| 구분 | 기업 |
|---|---|
| 선도 기업 | NVIDIA |
경쟁 해설[1].

## 6. 투자 판단 결과
| 항목 | 점수 |
|---|---|
판단 해설.

## 7. 사업 리스크
리스크 설명(확인 필요).

## 8. 한계점
한계 설명.

### 자동 검증 결과
- 수치 검증 결과

## REFERENCE
### 웹페이지
- [1] 매체A(2026). 기사A.
- [2] 매체B(2026). 기사B.
"""
SOURCES = [{"n": 7, "label": "웹페이지", "text": "매체A(2026). 기사A.", "title": "기사A", "raw": []},
           {"n": 9, "label": "웹페이지", "text": "매체B(2026). 기사B.", "title": "기사B", "raw": []}]


def test_previous_body_restores_citation_numbers_and_strips_added_parts():
    prev = reporter.previous_body({"report": PREV_REPORT}, SOURCES)
    assert "기술 내용[9]." in prev and "경쟁 해설[7]." in prev           # REFERENCE 번호 → 출처 목록 번호
    assert "자동 검증 결과" not in prev                                    # 검증 노드가 다시 붙인다
    assert "{{COMPETITOR_TABLE}}" in prev and "{{SCORE_TABLE}}" in prev and "| 선도 기업" not in prev


def test_flagged_chapters_from_instructions_values_and_marks():
    prev = reporter.previous_body({"report": PREV_REPORT}, SOURCES)
    assert reporter.flagged_chapters(prev, "- [편향 통제] 한계점 장에 인용 없음", []) == {"한계점", "사업 리스크"}
    assert "사업 아이디어·기술 성숙도" in reporter.flagged_chapters(prev, "1장 수치 오류", [])
    assert "사업 아이디어·기술 성숙도" in reporter.flagged_chapters(prev, "", ["기술 내용"])


def test_merge_keeps_unflagged_chapters_from_previous_report():
    prev = reporter.previous_body({"report": PREV_REPORT}, SOURCES)
    new = ("## SUMMARY\n새 결론.\n## 1. 사업 아이디어·기술 성숙도\n새 기술 문장(틀림).\n## 5. 경쟁 구도\n{{COMPETITOR_TABLE}}\n새 경쟁 해설[7].\n"
           "## 6. 투자 판단 결과\n{{SCORE_TABLE}}\n새 판단.\n## 7. 사업 리스크\n새 리스크[7].\n## 8. 한계점\n새 한계.\n")
    merged = reporter.merge_rewrite(new, prev, {"사업 리스크"})
    assert "기술 내용[9]." in merged and "새 기술 문장" not in merged      # 지적 안 된 장은 직전 내용
    assert "새 리스크[7]." in merged                                        # 지적된 장은 새 내용
    assert "새 경쟁 해설[7]." in merged and "새 판단." in merged            # 표 장은 항상 새 내용
    assert "**결론:** 투자." in merged and "한계 설명." in merged


def test_partial_rewrite_only_after_failed_check_without_research():
    base = {"report": PREV_REPORT, "report_version": 2}
    failed = {"report_version": 1, "passed": False, "fix_instructions": "- [중립성] 한계점 장 표현"}
    _, flagged = reporter.partial_rewrite_plan({**base, "eval_result": failed}, SOURCES)
    assert flagged == {"한계점", "사업 리스크"}
    _, after_research = reporter.partial_rewrite_plan({**base, "eval_result": {**failed, "evidence_gaps": {"team": "x"}}}, SOURCES)
    assert after_research == set()                                          # 재조사 후에는 전체를 새로 쓴다
    assert reporter.partial_rewrite_plan({"report": PREV_REPORT, "report_version": 1}, SOURCES) == ("", set())


def test_reporter_node_rewrites_only_flagged_chapter():
    prev = ("## SUMMARY\n추천 기업 없음.\n## 1. 평가 개요\n원래 개요.\n## 3. 사업 리스크\n원래 리스크.\n## 4. 한계점\n원래 한계.\n"
            "\n## REFERENCE\n- (사용한 출처 없음)\n")
    state = {**HOLD, "report": prev, "report_version": 2,
             "eval_result": {"report_version": 1, "passed": False, "fix_instructions": "- [중립성] 사업 리스크 장이 100자 미만"}}

    class RewriteLLM:
        def __init__(self):
            self.prompts = []

        def invoke(self, messages):
            self.prompts.append(messages[-1][1])
            return SimpleNamespace(content="## SUMMARY\n추천 기업 없음.\n## 1. 평가 개요\n바뀐 개요.\n"
                                           "## 3. 사업 리스크\n고친 리스크.\n## 4. 한계점\n바뀐 한계.\n")
    llm = RewriteLLM()
    report = reporter.reporter_node(state, llm=llm)["report"]
    assert "[부분 재작성]" in llm.prompts[0] and "{{SCORE_TABLE}}" in llm.prompts[0]
    assert "원래 개요." in report and "원래 한계." in report and "고친 리스크." in report
