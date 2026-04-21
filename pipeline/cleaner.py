"""Five-stage cleaning pipeline with audit logging."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any

import chardet

from config import pii_redaction_enabled


@dataclass
class StageLog:
    name: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class CleaningReport:
    stages: list[StageLog] = field(default_factory=list)
    final_text: str = ""
    quality_score: float = 0.0
    quality_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "stages": [{"name": s.name, "details": s.details} for s in self.stages],
            "quality_score": self.quality_score,
            "quality_notes": self.quality_notes,
            "char_count": len(self.final_text),
        }


_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff]")
_CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_EMAIL = re.compile(
    r"(?i)\b[a-z0-9._%+\-]+@[a-z0-9.\-]+\.[a-z]{2,}\b",
)
_PHONE = re.compile(
    r"(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{3}\)?[\s\-]?)\d{3}[\s\-]?\d{4}\b|\b\d{3}[\s\-]?\d{3}[\s\-]?\d{4}\b",
)
_SSN = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")


def _normalize_to_utf8_bytes(data: bytes) -> tuple[str, StageLog]:
    log = StageLog("encoding_fix", {"detected": None, "strategy": None})
    if data.startswith(b"\xef\xbb\xbf"):
        data = data[3:]
        log.details["bom"] = "utf-8-sig stripped"
    guess = chardet.detect(data) or {}
    enc = guess.get("encoding")
    confidence = guess.get("confidence")
    log.details["detected"] = enc
    log.details["confidence"] = confidence
    if enc:
        try:
            text = data.decode(enc, errors="strict")
            log.details["strategy"] = f"decode:{enc}"
            return text, log
        except UnicodeDecodeError:
            log.details["decode_error"] = enc
    try:
        text = data.decode("utf-8", errors="replace")
        log.details["strategy"] = "utf-8 replace"
        return text, log
    except Exception:
        text = data.decode("latin-1", errors="replace")
        log.details["strategy"] = "latin-1 replace"
        return text, log


def _ensure_text(raw: str | bytes) -> tuple[str, StageLog | None]:
    if isinstance(raw, str):
        return raw, None
    text, log = _normalize_to_utf8_bytes(raw)
    return text, log


def _text_cleanup(text: str) -> tuple[str, StageLog]:
    log = StageLog("text_cleanup", {"removed_zero_width": 0, "removed_controls": 0})
    before = len(text)
    text = _ZERO_WIDTH.sub("", text)
    log.details["removed_zero_width"] = before - len(text)
    text = _CTRL.sub("", text)
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    log.details["chars_in"] = before
    log.details["chars_out"] = len(text)
    return text, log


def _pii_redact(text: str, enabled: bool) -> tuple[str, StageLog]:
    log = StageLog(
        "pii_redaction",
        {"enabled": enabled, "emails": 0, "phones": 0, "ssn": 0},
    )
    if not enabled:
        return text, log

    def sub_count(pattern: re.Pattern, repl: str, s: str, key: str) -> str:
        n = len(pattern.findall(s))
        log.details[key] = n
        return pattern.sub(repl, s)

    text = sub_count(_EMAIL, "[EMAIL_REDACTED]", text, "emails")
    text = sub_count(_PHONE, "[PHONE_REDACTED]", text, "phones")
    text = sub_count(_SSN, "[SSN_REDACTED]", text, "ssn")
    return text, log


def _dedupe_lines(text: str) -> tuple[str, StageLog]:
    log = StageLog("deduplication", {"duplicate_chunks_removed": 0})
    lines = text.splitlines()
    seen: set[str] = set()
    kept: list[str] = []
    for line in lines:
        h = hashlib.sha256(line.encode("utf-8")).hexdigest()
        if h in seen:
            log.details["duplicate_chunks_removed"] = log.details.get("duplicate_chunks_removed", 0) + 1
            continue
        seen.add(h)
        kept.append(line)
    return "\n".join(kept), log


def _quality_score(text: str) -> tuple[float, str]:
    if not text:
        return 0.0, "empty"
    n = len(text)
    letters = sum(1 for c in text if c.isalpha())
    ratio = letters / max(n, 1)
    length_score = min(1.0, n / 200.0)
    alpha_score = min(1.0, ratio / 0.6) if ratio > 0 else 0.0
    score = round(0.5 * length_score + 0.5 * alpha_score, 3)
    notes = f"len={n}, alpha_ratio={ratio:.2f}"
    return score, notes


def clean_text(raw: str | bytes, *, dedupe_exact_chunks: bool = True) -> CleaningReport:
    """
    Run all stages on a single string or bytes blob.
    For per-chunk dedup + quality, use ``clean_for_index`` on each chunk after chunking.
    """
    report = CleaningReport()
    text, enc_log = _ensure_text(raw)
    if enc_log:
        report.stages.append(enc_log)

    text, tlog = _text_cleanup(text)
    report.stages.append(tlog)

    text, plog = _pii_redact(text, pii_redaction_enabled())
    report.stages.append(plog)

    if dedupe_exact_chunks:
        text, dlog = _dedupe_lines(text)
        report.stages.append(dlog)

    score, notes = _quality_score(text)
    report.stages.append(StageLog("quality_check", {"score": score, "notes": notes}))
    report.quality_score = score
    report.quality_notes = notes
    report.final_text = text
    return report
