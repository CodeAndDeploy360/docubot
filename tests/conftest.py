import pytest


@pytest.fixture(autouse=True)
def fresh_chroma(monkeypatch, tmp_path):
    """Isolate Chroma state and reset the persistent client between tests."""
    import rag.store as st

    st._client = None
    root = tmp_path / "chroma_data"
    root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("DOCUBOT_CHROMA_PATH", str(root))
