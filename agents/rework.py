"""Supervisor 재작업 요청을 분석 에이전트가 읽을 때 쓰는 공통 도우미.

supervisor는 근거가 부족한 관점에 state["rework_request"][노드] = {"reason", "focus"}를 남기고 그 노드만 다시 부른다.
에이전트는 검색어 고정·seed 고정이라 같은 입력이면 같은 결과가 나오므로, 재작업 때는 첫 실행과 다른 검색어를 쓰고
이전 결과에서 이미 확인한 사실은 유지한 채 새로 찾은 것만 더한다.
"""


def rework_of(state: dict, node: str) -> dict | None:
    """이번 실행이 재작업이면 {"reason", "focus"}, 아니면 None.
    실행 오류 뒤 재시도(focus가 retry뿐)는 보완할 내용이 없으므로 첫 실행과 같게 돌도록 None으로 본다.
    """
    request = (state.get("rework_request") or {}).get(node)
    if not request or set(request.get("focus") or []) <= {"retry"}:
        return None
    return request


def previous(state: dict, key: str) -> dict | None:
    """재작업 전 결과. 실행 오류로 비어 있던 결과({"failed": True})는 유지할 것이 없으므로 None."""
    prev = state.get(key)
    return prev if prev and not prev.get("failed") else None


def union(old: list, new: list, key=lambda x: x) -> list:
    """순서를 지키며 합친다. 같은 key는 먼저 나온 것(이전 결과)을 남긴다."""
    seen, out = set(), []
    for x in [*old, *new]:
        k = key(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out
