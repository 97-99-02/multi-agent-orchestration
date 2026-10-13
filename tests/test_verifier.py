"""사실 검증 수치 읽기 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_verifier.py -q"""
import pytest

from agents import verifier


@pytest.mark.parametrize("text, value, unit", [
    ("3,400백만달러", 3.4e9, "달러"),       # 원문 표기. 34억 달러
    ("14백만달러", 1.4e7, "달러"),
    ("5천만 원", 5e7, "원"),                # '천'만 읽으면 5000이 된다
    ("3.4억 달러", 3.4e8, "달러"),
    ("2천 명", 2e3, "명"),
])
def test_parse_quantities_reads_million_units(text, value, unit):
    [q] = verifier.parse_quantities(text)
    assert (q["value"], q["unit"]) == (value, unit)


def test_wrong_unit_conversion_is_a_mismatch():
    # 보고서가 원문 '3,400백만달러'를 '3.4억 달러'로 잘못 바꿔 덧붙인 경우 (실제로는 34억 달러)
    state = {"company": {"name": "A사"}, "sources": [],
             "market_analysis": {"figures": [{"kind": "시장규모", "value": "3,400백만달러", "year": ""}]}}
    evidence = verifier.build_evidence(state)
    mismatches, _ = verifier.find_quantity_mismatches("## 2. 시장성\nCXL 시장은 3,400백만달러(3.4억 달러)로 전망된다.\n", evidence)
    assert [m["value"] for m in mismatches] == ["3.4억 달러"]
    ok, _ = verifier.find_quantity_mismatches("## 2. 시장성\nCXL 시장은 34억 달러로 전망된다.\n", evidence)
    assert ok == []
