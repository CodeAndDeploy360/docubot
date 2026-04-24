"""Signed cookie payload for “stay signed in” across full page refresh (Streamlit session_state alone does not)."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time

import config

_MAX_AGE_SEC = 30 * 24 * 60 * 60


def create_session_cookie_value(user_id: str, email: str) -> str:
    exp = int(time.time()) + _MAX_AGE_SEC
    body = json.dumps(
        {"uid": user_id, "e": (email or "").strip().lower(), "exp": exp},
        separators=(",", ":"),
        sort_keys=True,
    )
    b64 = base64.urlsafe_b64encode(body.encode()).decode().rstrip("=")
    sig = hmac.new(
        config.session_signing_secret(),
        b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()
    return f"{b64}.{sig}"


def verify_session_cookie_value(raw: str) -> tuple[str, str] | None:
    s = (raw or "").strip()
    if not s or "." not in s:
        return None
    b64, sig = s.rsplit(".", 1)
    if hmac.new(
        config.session_signing_secret(),
        b64.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest() != sig:
        return None
    try:
        pad = (-len(b64)) % 4
        payload = b64 + ("=" * pad if pad else "")
        data = json.loads(base64.urlsafe_b64decode(payload))
        if int(data.get("exp", 0)) < time.time():
            return None
        uid, em = str(data.get("uid", "")), str(data.get("e", ""))
        if not uid or not em:
            return None
        return uid, em
    except (json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None
