"""경쟁사 비교 에이전트 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_competitor.py -q"""
import obs
from agents import competitor

# 4회차 실제 출처(디일렉 대표 인터뷰)를 줄인 원문: 대표가 한 말이 '가장 위협적인 경쟁사로 평가된다'로 바뀌었었다
INTERVIEW = {"title": "모빌린트, 올해 매출 가시화 - 디일렉", "url": "https://a.com/1",
             "published_date": "Tue, 04 Mar 2025 00:00:00 GMT",
             "content": "신 대표는 에지 AI 분야에서는 퀄컴이 가장 위협이 될 회사라고 보고 있다면서도 충분히 경쟁할 수 있다고 자신했다. "
                        "모빌린트는 NPU용 SDK를 제공한다. 누적 투자금은 700억 원이다."}
NEWS = {"title": "엔비디아, 젯슨 오린 출시 - 매체B", "url": "https://b.com/2", "published_date": "Mon, 12 May 2026 00:00:00 GMT",
        "content": "엔비디아는 2023년 젯슨 오린을 출시했다. 삼성전자는 올해 4분기 CXL 3.1 양산을 추진한다."}
TESTER = {"title": "엑시콘 CXL 테스터 - 매체C", "url": "https://c.com/3", "published_date": "Mon, 12 May 2026 00:00:00 GMT",
          "content": "엑시콘은 CXL 2.0 테스터를 공급한다."}
RESULTS = [INTERVIEW, NEWS, TESTER]


def claim(text, evidence_id, basis="사실", speaker="제3자", **kw):
    return {"text": text, "basis": basis, "speaker": speaker, "comparable": False, "evidence_id": evidence_id, **kw}


def test_years_and_numbers_must_be_in_source():
    body = NEWS["content"]
    assert not competitor.facts_in_source("삼성전자는 2025년 4분기 CXL 3.1을 양산할 예정이다", body, "2026-05-12")
    assert competitor.facts_in_source("삼성전자는 2026년 4분기 CXL 3.1 양산을 추진한다", body, "2026-05-12")   # '올해' = 게시 연도
    assert not competitor.facts_in_source("엔비디아는 2023년 젯슨 오린을 출시했고 성능은 275 TOPS다", body, "2026-05-12")
    assert competitor.facts_in_source("’23년 출시", "’23년 출시")


def test_opinions_funding_and_bad_numbers_are_dropped():
    claims = [claim("퀄컴은 가장 위협적인 경쟁사다", 0, basis="의견", speaker="대상 기업"),
              claim("누적 투자금 700억 원", 0, speaker="대상 기업", kind="고객·양산"),
              claim("NPU용 SDK를 제공한다", 0, speaker="대상 기업", kind="소프트웨어 개발 환경"),
              claim("삼성전자는 2025년 4분기 CXL 3.1을 양산할 예정이다", 1),
              claim("근거 번호가 틀린 주장", 9)]
    kept, dropped = competitor.check_claims(claims, RESULTS, strengths=True)
    assert [c["text"] for c in kept] == ["NPU용 SDK를 제공한다"]
    assert kept[0]["url"] == INTERVIEW["url"] and kept[0]["date"] == "2025-03-04" and "evidence_id" not in kept[0]
    assert [d.split(":")[0] for d in dropped] == ["의견", "투자 정보", "숫자·연도가 원문에 없음", "근거 번호 없음"]


def test_claim_text_names_who_announced_it():
    assert competitor.claim_text({"text": "SDK 제공", "speaker": "대상 기업"}, "모빌린트") == "SDK 제공 (모빌린트 발표)"
    assert competitor.claim_text({"text": "점유율 15%", "speaker": "경쟁사"}, "모빌린트", "퀄컴") == "점유율 15% (퀄컴 발표)"
    assert competitor.claim_text({"text": "MLPerf 결과", "speaker": "제3자"}, "모빌린트") == "MLPerf 결과"


class FakeLLM:
    """경쟁사 분석·분야 판정 호출을 흉내 낸다. 엑시콘(검사장비)은 분야 판정에서 빠진다."""
    def __init__(self, **_):
        self.schema = None

    def with_structured_output(self, schema):
        self.schema = schema
        return self

    def with_retry(self, **_):
        return self

    def invoke(self, _):
        if self.schema is competitor.SegmentLabels:
            label = competitor.SegmentLabel
            return competitor.SegmentLabels(labels=[label(index=0, in_segment=True, tier="선도 기업", basis="상장사"),
                                                    label(index=1, in_segment=True, tier="선도 기업", basis="상장사"),
                                                    label(index=2, in_segment=False, tier="선도 기업", basis="장비 공급")])
        comps = [competitor.Competitor(name="Qualcomm", country="미국", product="Snapdragon", summary="모바일 AP·NPU",
                                       claims=[competitor.Claim(**claim("퀄컴은 가장 위협적인 경쟁사로 평가된다", 0,
                                                                        basis="의견", speaker="대상 기업"))]),
                 competitor.Competitor(name="NVIDIA", country="미국", product="Jetson Orin", summary="엣지 AI 모듈",
                                       claims=[competitor.Claim(**claim("엔비디아는 2023년 젯슨 오린을 출시했다", 1))]),
                 competitor.Competitor(name="엑시콘", country="한국", product="CXL 테스터", summary="검사장비", claims=[])]
        return competitor.CompetitorAnalysis(
            competitors=comps,
            differentiation=[competitor.Strength(**claim("누적 투자금 700억 원", 0, speaker="대상 기업", kind="고객·양산")),
                             competitor.Strength(**claim("NPU용 SDK를 제공한다", 0, speaker="대상 기업", kind="소프트웨어 개발 환경"))],
            competitive_risks=["퀄컴의 엣지 AI 시장 진입"], not_comparable=["TOPS/W: 측정 조건이 다름"], evidence_ids=[1])


def test_competitor_node_keeps_only_checked_claims(monkeypatch):
    monkeypatch.setattr(competitor, "ChatOpenAI", FakeLLM)
    monkeypatch.setattr(competitor, "search_many", lambda *a, **k: [dict(r) for r in RESULTS])
    state = {"trace_id": "t-competitor", "company": {"name": "모빌린트", "segment": "엣지", "product": "ARIES"}}
    out = competitor.competitor_node(state)
    analysis = out["competitor_analysis"]
    by_name = {c["name"]: c for c in analysis["competitors"]}
    assert list(by_name) == ["Qualcomm", "NVIDIA"] and analysis["excluded_other_segment"] == ["엑시콘"]
    assert by_name["Qualcomm"]["claims"] == [] and by_name["Qualcomm"]["comparison"] == competitor.NO_COMPARISON
    assert by_name["NVIDIA"]["comparison"] == "엔비디아는 2023년 젯슨 오린을 출시했다"
    assert [d["text"] for d in analysis["differentiation"]] == ["NPU용 SDK를 제공한다"]
    assert analysis["not_comparable"] == ["TOPS/W: 측정 조건이 다름"]
    assert sorted(s["url"] for s in out["sources"]) == [INTERVIEW["url"], NEWS["url"]]   # 버린 주장만 있던 출처는 안 남는다
    [event] = obs.read_events("t-competitor", "diagnostics")
    assert any(d.startswith("Qualcomm 의견") for d in event["data"]["dropped_claims"])
