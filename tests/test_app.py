"""실행 설정 테스트 (API 키 없이 실행): uv run --with pytest python -m pytest tests/test_app.py -q

가짜 노드 그래프로 trace_id 연결(State·LangSmith metadata·결정 로그·thread_id), 체크포인터 저장,
예외 후 마지막 체크포인트에서 이어가기를 확인한다.
"""
import pytest
from langgraph.checkpoint.memory import InMemorySaver

import app as runner
import obs
from config import MAX_RESUME, MAX_STEPS, RECURSION_LIMIT
from graph import build_graph
from test_graph_flow import Scenario


def fake_app(scenario, nodes=None):
    return build_graph({**scenario.nodes(), **(nodes or {})}, checkpointer=InMemorySaver())


def test_config_uses_trace_id_everywhere():
    config = runner.make_config("abcdef1234567890")
    assert config["metadata"] == {"trace_id": "abcdef1234567890"}
    assert config["configurable"] == {"thread_id": "abcdef1234567890"}
    assert config["run_name"].endswith("abcdef12")
    assert config["recursion_limit"] == RECURSION_LIMIT > 2 * MAX_STEPS


def test_run_links_state_log_and_checkpoint():
    s = Scenario(["A사"], {"A사": "투자"})
    graph = fake_app(s)
    state, config = runner.run(graph, trace_id="t-123")
    assert state["trace_id"] == "t-123"
    assert obs.read_events("t-123", kind="decision")              # 결정 로그 파일 이름 = trace_id
    snapshot = graph.get_state(config)                            # 체크포인터에 thread_id = trace_id로 저장됨
    assert snapshot.values["report"] == state["report"] and snapshot.next == ()


def test_run_resumes_from_last_checkpoint_after_exception():
    s = Scenario(["A사"], {"A사": "투자"})
    calls = {"reporter": 0}
    real_reporter = s.nodes()["reporter"]

    def flaky_reporter(state):
        calls["reporter"] += 1
        if calls["reporter"] == 1:
            raise ConnectionError("OpenAI 일시 오류")
        return real_reporter(state)

    state, _ = runner.run(fake_app(s, {"reporter": flaky_reporter}), trace_id="t-resume")
    assert state["report"] == "보고서 v1"
    assert s.calls.count("explorer") == 1 and s.calls.count("judge") == 1   # 앞 단계는 다시 하지 않음
    [event] = obs.read_events("t-resume", kind="resume")
    assert event["next"] == ["reporter"] and "OpenAI 일시 오류" in event["error"]


def test_run_gives_up_after_max_resume():
    s = Scenario(["A사"], {"A사": "투자"})

    def broken_reporter(state):
        raise ConnectionError("계속 실패")

    with pytest.raises(ConnectionError):
        runner.run(fake_app(s, {"reporter": broken_reporter}), trace_id="t-fail")
    assert len(obs.read_events("t-fail", kind="resume")) == MAX_RESUME


# ---------- 프로그램을 다시 켜서 이어가기 (SqliteSaver) ----------
def test_resume_after_restart_from_sqlite_checkpoint(tmp_path):
    s = Scenario(["A사"], {"A사": "투자"})
    db = tmp_path / "checkpoints.sqlite"

    def broken_reporter(state):
        raise ConnectionError("OpenAI 장애")

    first = build_graph({**s.nodes(), "reporter": broken_reporter}, checkpointer=runner.make_checkpointer(db))
    with pytest.raises(ConnectionError):
        runner.run(first, trace_id="t-restart")
    # 새 프로그램: 그래프·체크포인터를 새로 만들고 같은 DB 파일에서 이어간다
    second = build_graph(s.nodes(), checkpointer=runner.make_checkpointer(db))
    state, _ = runner.run(second, trace_id="t-restart", resume=True)
    assert state["report"] == "보고서 v1"
    assert s.calls.count("explorer") == 1 and s.calls.count("judge") == 1    # 앞 단계는 다시 하지 않음
    assert any(e["error"].startswith("(프로그램을 다시 켜서") for e in obs.read_events("t-restart", kind="resume"))


def test_resume_unknown_or_finished_run(tmp_path):
    s = Scenario(["A사"], {"A사": "투자"})
    db = tmp_path / "checkpoints.sqlite"
    graph = build_graph(s.nodes(), checkpointer=runner.make_checkpointer(db))
    with pytest.raises(SystemExit):
        runner.run(graph, trace_id="없는-실행", resume=True)
    runner.run(graph, trace_id="t-done")
    calls = len(s.calls)
    state, _ = runner.run(graph, trace_id="t-done", resume=True)
    assert state["report"] and len(s.calls) == calls                      # 끝난 실행은 다시 돌리지 않음
