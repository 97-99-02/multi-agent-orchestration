"""결정 로그 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_obs.py -q"""
import threading

import pytest

import obs


@pytest.fixture(autouse=True)
def tmp_log_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(obs, "LOG_DIR", tmp_path / "decisions")


def test_decision_is_written_with_trace_id_and_reason():
    obs.log_decision("t1", "supervisor", ["market"], "market: 세부 분야 성장률 없음 → 재작업 1/1", step=3)
    [event] = obs.read_events("t1")
    assert event["trace_id"] == "t1"
    assert event["kind"] == "decision"
    assert event["decision"] == ["market"]
    assert event["reason"].startswith("market:")
    assert event["step"] == 3
    assert event["ts"]


def test_events_are_appended_and_filtered_by_kind():
    obs.log_decision("t1", "supervisor", ["explorer"], "평가 중인 기업 없음")
    obs.log_diagnostics("t1", "tech_summary", "A사", {"web_queries": ["A사 NPU"]})
    obs.log_decision("t1", "supervisor", ["judge"], "4개 관점 충분")
    assert [e["decision"] for e in obs.read_events("t1", kind="decision")] == [["explorer"], ["judge"]]
    assert obs.read_events("t1", kind="diagnostics")[0]["data"]["web_queries"] == ["A사 NPU"]


def test_runs_are_separated_by_trace_id():
    obs.log_decision("t1", "supervisor", ["explorer"], "a")
    obs.log_decision("t2", "supervisor", ["explorer"], "b")
    assert len(obs.read_events("t1")) == 1
    assert len(obs.read_events("t2")) == 1


def test_missing_trace_id_goes_to_no_trace_file():
    obs.log_decision(None, "supervisor", [], "x")
    assert obs.log_path(None).name == "no-trace.jsonl"
    assert len(obs.read_events(None)) == 1


def test_parallel_writes_do_not_break_lines():
    # 병렬 분석 노드 4개가 동시에 진단을 남기는 경우
    def write(i):
        for j in range(25):
            obs.log_diagnostics("t1", f"node{i}", "A사", {"j": j, "text": "가" * 200})

    threads = [threading.Thread(target=write, args=(i,)) for i in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(obs.read_events("t1")) == 100
