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
def test_toc_chapters_1_to_4_carry_the_four_perspectives():
    titles = toc_titles()
    for i, view in enumerate(quality.PERSPECTIVES, start=1):
        assert titles[i].startswith(f"{i}.") and view in titles[i], (view, titles[i])


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
    assert "관점 커버리지" in report
    assert quality.add_quality_note(report, out["eval_result"]["items"]) == report   # 두 번 넣지 않음


def test_no_quality_note_before_loop_limit():
    out = quality.quality_eval_node({**INVEST, "quality_loops": MAX_QUALITY_LOOPS - 1}, judge_llm=False)
    assert "report" not in out
