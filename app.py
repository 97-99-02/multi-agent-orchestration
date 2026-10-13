"""실행: uv run python app.py
이어서 실행: uv run python app.py --resume <trace_id>

실행마다 trace_id를 만들어 다음 네 곳에 같은 값을 넣는다 (상관).
- State의 trace_id
- LangSmith run metadata (run_name 끝 8자리도 같음)
- 결정 로그 파일 outputs/decisions/{trace_id}.jsonl
- 체크포인터 thread_id

체크포인터(SqliteSaver)는 단계마다 State를 outputs/checkpoints.sqlite에 저장한다 (재개/복구).
- 실행 중 예외로 멈추면 같은 실행 안에서 마지막 체크포인트부터 이어간다 (최대 MAX_RESUME회)
- 그래도 멈추거나 프로그램이 꺼지면, 다시 켜서 --resume <trace_id>로 마지막 체크포인트부터 이어간다
"""
import argparse
import json
import logging
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver

from agents.reporter import export_report_pdf
from config import MAX_RESUME, RECURSION_LIMIT
from graph import build_graph
from obs import log_event, log_path

load_dotenv()
logger = logging.getLogger(__name__)


CHECKPOINT_DB = Path("outputs") / "checkpoints.sqlite"


def make_checkpointer(path: Path = CHECKPOINT_DB) -> SqliteSaver:
    path.parent.mkdir(parents=True, exist_ok=True)
    # 병렬 분석 노드는 다른 스레드에서 체크포인트를 저장하므로 같은 스레드 제한을 끈다 (SqliteSaver가 잠금으로 순서를 지킨다)
    return SqliteSaver(sqlite3.connect(str(path), check_same_thread=False))


def make_config(trace_id: str) -> dict:
    return {
        "recursion_limit": RECURSION_LIMIT,
        "run_name": f"invest-supervisor-{trace_id[:8]}",
        "tags": ["agent-assignment", "supervisor"],
        "metadata": {"trace_id": trace_id},
        "configurable": {"thread_id": trace_id},
    }


def run(app=None, trace_id: str | None = None, resume: bool = False) -> tuple[dict, dict]:
    """그래프를 끝까지 실행하고 (최종 State, config)를 돌려준다.
    resume=True면 저장된 trace_id 실행을 마지막 체크포인트부터 이어간다 (프로그램을 다시 켠 경우)."""
    trace_id = trace_id or uuid.uuid4().hex
    app = app or build_graph(checkpointer=make_checkpointer())
    config = make_config(trace_id)
    inputs = {"trace_id": trace_id}
    if resume:
        snapshot = app.get_state(config)
        if not snapshot.values:
            raise SystemExit(f"저장된 실행이 없습니다: {trace_id}")
        if not snapshot.next:
            logger.info("[실행] %s는 이미 끝난 실행입니다", trace_id)
            return snapshot.values, config
        logger.info("[실행] %s를 마지막 체크포인트(다음 노드 %s)부터 이어갑니다", trace_id, snapshot.next)
        log_event(trace_id, "resume", "app", error="(프로그램을 다시 켜서 이어감)", next=list(snapshot.next))
        inputs = None
    for attempt in range(MAX_RESUME + 1):
        try:
            return app.invoke(inputs, config), config
        except Exception as e:  # noqa: BLE001 - 대체 결과가 없는 노드(reporter 등)의 예외
            if attempt == MAX_RESUME:
                raise
            snapshot = app.get_state(config)
            logger.warning("[실행] %s → 마지막 체크포인트(다음 노드 %s)부터 이어갑니다", e, snapshot.next)
            log_event(trace_id, "resume", "app", error=f"{type(e).__name__}: {e}"[:300], next=list(snapshot.next))
            inputs = None   # None으로 부르면 저장된 체크포인트부터 이어간다
    raise RuntimeError("unreachable")


def main():
    parser = argparse.ArgumentParser(description="AI 반도체 스타트업 투자 평가 (Supervisor 패턴)")
    parser.add_argument("--resume", metavar="TRACE_ID", help="멈춘 실행을 마지막 체크포인트부터 이어간다")
    args = parser.parse_args()
    trace_id = args.resume or uuid.uuid4().hex
    try:
        state, config = run(trace_id=trace_id, resume=bool(args.resume))
    except Exception:
        print(f"\n실행이 멈췄습니다. 이어서 실행: uv run python app.py --resume {trace_id}")
        raise
    trace_id = config["metadata"]["trace_id"]
    out_dir = Path("outputs")
    out_dir.mkdir(exist_ok=True)
    path = out_dir / f"report_{datetime.now():%Y%m%d_%H%M}_{trace_id[:8]}.md"
    path.write_text(state["report"], encoding="utf-8")
    print(f"보고서 저장: {path}")
    # 최종 State를 남겨 두면 PDF 모양만 고칠 때 다시 실행하지 않고 같은 결과로 다시 만들 수 있다
    state_path = out_dir / "runs" / f"state_{trace_id}.json"
    state_path.parent.mkdir(exist_ok=True)
    state_path.write_text(json.dumps(state, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"최종 State 저장: {state_path}")
    pdf_path, pages = export_report_pdf(state["report"], state=state)
    print(f"제출용 PDF 저장: {pdf_path} ({pages}쪽)")
    print(f"trace_id: {trace_id} (LangSmith run: {config['run_name']})")
    print(f"결정 로그: {log_path(trace_id)}")
    print(f"supervisor 실행 {state.get('step_count')}회, 마지막 결정: {state.get('last_decision')}")
    print(f"재작업: {state.get('rework_count') or '없음'}, 보고서 버전: {state.get('report_version')}, "
          f"사실 검증 재작성 {state.get('retry_count', 0)}회, 품질 재작성 {state.get('quality_loops', 0)}회")
    print(f"평가 후보: {state['candidates']}")
    print(f"보류/제외: {[r['company'] for r in state.get('rejected', [])]}")
    if state.get("agent_errors"):
        print(f"노드 오류: {[(e['node'], e['error'][:60]) for e in state['agent_errors']]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for name in ("httpx", "httpx2"):   # API 요청마다 찍히는 로그는 끈다
        logging.getLogger(name).setLevel(logging.WARNING)
    main()
