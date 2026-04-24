import pytest


@pytest.fixture(autouse=True)
def fresh_chroma(monkeypatch, tmp_path):
    """Isolate per-user Chroma under tmp_path and fix a test user id."""
    import rag.store as st

    st.reset_for_tests()
    monkeypatch.setenv("DOCUBOT_DATA_DIR", str(tmp_path))
    st.set_active_user("00000000-0000-0000-0000-000000000001")
