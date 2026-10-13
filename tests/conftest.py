"""모든 테스트 공통 설정."""
import pytest

import obs


@pytest.fixture(autouse=True)
def isolated_decision_log(tmp_path, monkeypatch):
    """노드를 직접 부르는 테스트도 결정 로그를 outputs/decisions가 아니라 임시 폴더에 쓰게 한다."""
    monkeypatch.setattr(obs, "LOG_DIR", tmp_path / "decisions")
