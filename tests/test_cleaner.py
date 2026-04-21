"""Tests for pipeline.cleaner (spec: data cleaning pipeline + PII)."""

from __future__ import annotations

import importlib
from pathlib import Path

_TEST_DATA = Path(__file__).resolve().parent.parent / "test_data"


def _reload_cleaner(monkeypatch, *, pii: str) -> None:
    monkeypatch.setenv("DOCUBOT_PII_REDACTION_ENABLED", pii)
    import config
    import pipeline.cleaner as cl

    importlib.reload(config)
    importlib.reload(cl)


def test_pii_redacts_email_phone_ssn(monkeypatch):
    _reload_cleaner(monkeypatch, pii="true")
    import pipeline.cleaner as cl

    raw = "Reach us at alice@example.com or 415-555-0199. SSN 123-45-6789."
    report = cl.clean_text(raw, dedupe_exact_chunks=False)
    assert "[EMAIL_REDACTED]" in report.final_text
    assert "[PHONE_REDACTED]" in report.final_text
    assert "[SSN_REDACTED]" in report.final_text


def test_pii_can_disable(monkeypatch):
    _reload_cleaner(monkeypatch, pii="false")
    import pipeline.cleaner as cl

    raw = "alice@example.com"
    report = cl.clean_text(raw, dedupe_exact_chunks=False)
    assert "alice@example.com" in report.final_text


def test_text_cleanup_strips_control_chars(monkeypatch):
    _reload_cleaner(monkeypatch, pii="true")
    import pipeline.cleaner as cl

    raw = "Hello\x00\x01 world\n\n\nthere"
    report = cl.clean_text(raw, dedupe_exact_chunks=False)
    assert "Hello" in report.final_text
    assert "world" in report.final_text


def test_audit_stages_recorded(monkeypatch):
    _reload_cleaner(monkeypatch, pii="true")
    import pipeline.cleaner as cl

    report = cl.clean_text("hello", dedupe_exact_chunks=False)
    names = [s.name for s in report.stages]
    assert "text_cleanup" in names
    assert "pii_redaction" in names
    assert "quality_check" in names


def test_clean_text_accepts_utf8_bytes(monkeypatch):
    _reload_cleaner(monkeypatch, pii="true")
    import pipeline.cleaner as cl

    report = cl.clean_text("café data".encode("utf-8"), dedupe_exact_chunks=False)
    assert "caf" in report.final_text
    stage_names = [s.name for s in report.stages]
    assert "encoding_fix" in stage_names


def test_deduplication_removes_duplicate_lines(monkeypatch):
    _reload_cleaner(monkeypatch, pii="true")
    import pipeline.cleaner as cl

    raw = "same line\nsame line\nunique\n"
    report = cl.clean_text(raw, dedupe_exact_chunks=True)
    assert report.final_text.count("same line") == 1
    assert any(s.name == "deduplication" for s in report.stages)


def test_report_to_dict(monkeypatch):
    _reload_cleaner(monkeypatch, pii="true")
    import pipeline.cleaner as cl

    d = cl.clean_text("abc", dedupe_exact_chunks=False).to_dict()
    assert "stages" in d and "quality_score" in d
    assert d["char_count"] == 3


def test_pii_test_txt_fixture_redacts(monkeypatch):
    """Spec demo file ``test_data/pii_test.txt``: emails, phones, SSN-style values masked."""
    _reload_cleaner(monkeypatch, pii="true")
    import pipeline.cleaner as cl

    raw = (_TEST_DATA / "pii_test.txt").read_text(encoding="utf-8")
    report = cl.clean_text(raw, dedupe_exact_chunks=False)
    t = report.final_text
    assert "jane.doe@example.com" not in t
    assert "[EMAIL_REDACTED]" in t
    assert "[PHONE_REDACTED]" in t
    assert "[SSN_REDACTED]" in t
