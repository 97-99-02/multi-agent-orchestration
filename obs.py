"""관측성: 결정 로그와 진단 정보를 State 밖 파일에 남긴다.

State에는 마지막 결정 한 줄(last_decision)만 두고, 결정과 사유 전체는 outputs/decisions/{trace_id}.jsonl에
한 줄씩 쌓는다. 같은 trace_id를 LangSmith metadata와 체크포인터 thread_id에도 넣으므로
State·트레이스·결정 로그를 이 키 하나로 맞춰 볼 수 있다 (상관).
체크포인트마다 저장되는 State에 로그 본문을 넣지 않으니 State가 실행 길이만큼 커지지 않는다 (지속성 비용).
"""
import json
import threading
from datetime import datetime
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "outputs" / "decisions"
_LOCK = threading.Lock()   # 병렬 분석 노드가 같은 파일에 동시에 쓸 수 있다


def log_path(trace_id: str | None) -> Path:
    return LOG_DIR / f"{trace_id or 'no-trace'}.jsonl"


def log_event(trace_id: str | None, kind: str, node: str, **fields) -> dict:
    """{trace_id, ts, kind, node, ...}를 한 줄 추가하고 그 기록을 돌려준다."""
    record = {"trace_id": trace_id, "ts": datetime.now().isoformat(timespec="seconds"),
              "kind": kind, "node": node, **fields}
    line = json.dumps(record, ensure_ascii=False, default=str)
    with _LOCK:
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        with log_path(trace_id).open("a", encoding="utf-8") as f:
            f.write(line + "\n")
    return record


def log_decision(trace_id: str | None, node: str, decision, reason: str, **extra) -> dict:
    """라우팅·판정 결정과 사유를 남긴다. decision은 다음 노드 목록이나 판정 값이다.
    reason은 사람이 읽을 수 있게 쓴다. 예: "market: 세부 분야 성장률 없음 → 재작업 1/1"
    """
    return log_event(trace_id, "decision", node, decision=decision, reason=reason, **extra)


def log_diagnostics(trace_id: str | None, node: str, company: str, data: dict) -> dict:
    """에이전트 내부 진단(검색어, 거절한 주장 수 등)을 State 대신 여기에 남긴다."""
    return log_event(trace_id, "diagnostics", node, company=company, data=data)


def read_events(trace_id: str | None, kind: str | None = None) -> list[dict]:
    path = log_path(trace_id)
    if not path.exists():
        return []
    events = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [e for e in events if kind is None or e["kind"] == kind]
