"""보고서 품질 평가 노드 (담당: 김다은)

평가 항목: Groundedness, 중립성, 편향 통제, 관점 커버리지 (코드 검사 + LLM Judge)
출력: {"eval_result": {"passed", "report_version", "items": {항목: {"passed", "reasons"}}, "fix_instructions"}}

[임시] 그래프 연결용 자리표시. 항상 통과로 돌려준다. 8단계에서 실제 평가로 교체한다.
"""


def quality_eval_node(state: dict) -> dict:
    return {"eval_result": {"passed": True, "report_version": state.get("report_version"),
                            "items": {}, "fix_instructions": "", "placeholder": True}}
