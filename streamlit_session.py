"""Persist sign-in across full page refresh: server session + URL param, with cookie fallback."""

from __future__ import annotations

import json
import secrets
import time
import urllib.parse
from typing import Any

import streamlit as st

from auth import db
from auth.session_token import create_session_cookie_value, verify_session_cookie_value

# Survives F5: full URL is reloaded; `st.context.cookies` alone is unreliable on some clients.
SESSION_QP = "docubot_sid"
# One-shot: next script run must not re-login from cookie/URL (cookie clear may land one frame late).
LOGOUT_FLAG = "_docubot_signed_out"
# One URL+cookie restore pass per browser session (st.session_state). Without this, every widget
# rerun re-applies ?docubot_sid= and signs the user in while they use e.g. Resend / Forgot password.
SESSION_RESTORE_RAN = "_docubot_session_restore_ran"

COOKIE_NAME = "docubot_s"
_SESSION_TTL_SEC = 30 * 24 * 60 * 60
_MAX_AGE = _SESSION_TTL_SEC


def _context_cookies() -> Any | None:
    try:
        return st.context.cookies
    except Exception:
        return None


def _qp_get(name: str) -> str | None:
    try:
        v = st.query_params.get(name)
    except Exception:
        return None
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return str(v[0]).strip() if v else None
    s = str(v).strip()
    return s if s else None


def _qp_pop(name: str) -> None:
    try:
        st.query_params.pop(name, None)  # type: ignore[call-overload]
    except (TypeError, KeyError, AttributeError):
        try:
            del st.query_params[name]  # type: ignore[misc]
        except (TypeError, KeyError, AttributeError):
            pass


def read_session_cookie() -> str | None:
    c = _context_cookies()
    if c is None:
        return None
    v: Any = None
    try:
        v = c[COOKIE_NAME]  # type: ignore[index]
    except (KeyError, TypeError):
        try:
            for k, val in c.items():  # type: ignore[union-attr]
                if str(k).lower() == COOKIE_NAME.lower():
                    v = val
                    break
        except Exception:
            return None
    if v is None or v == "":
        return None
    return urllib.parse.unquote(str(v))


def _inject_set_session_cookie(value: str) -> None:
    enc = urllib.parse.quote(value, safe="")
    secure = ""
    try:
        if str(st.context.url or "").lower().startswith("https:"):
            secure = "; Secure"
    except Exception:
        pass
    q = f"{COOKIE_NAME}={enc}; path=/; max-age={_MAX_AGE}; SameSite=Lax{secure}"
    st.html(
        f"<script>document.cookie = {json.dumps(q)};</script>",
        unsafe_allow_javascript=True,
    )


def _inject_clear_session_cookie() -> None:
    q = f"{COOKIE_NAME}=; path=/; max-age=0; SameSite=Lax"
    st.html(
        f"<script>document.cookie = {json.dumps(q)};</script>",
        unsafe_allow_javascript=True,
    )


def _apply_user_row(row: dict[str, Any]) -> None:
    st.session_state["user_id"] = str(row["id"])
    st.session_state["user_email"] = str(row["email"])


def _write_cookie_for_user(user_id: str, email: str) -> None:
    try:
        token = create_session_cookie_value(user_id, email)
    except Exception:
        return
    _inject_set_session_cookie(token)


def persist_session_after_login(user_id: str, email: str) -> None:
    """Call after successful login. Puts a session id in the URL (reliable) and sets a signed cookie (extra)."""
    st.session_state.pop(SESSION_RESTORE_RAN, None)
    sid = secrets.token_urlsafe(32)
    db.insert_user_session(sid, user_id, time.time() + _SESSION_TTL_SEC)
    try:
        st.query_params[SESSION_QP] = sid
    except Exception:
        pass
    _write_cookie_for_user(user_id, email)


def try_restore_user_session() -> None:
    """
    Rehydrate user_id from URL session id, else from signed cookie.
    Run once after st.set_page_config (before auth gate).
    """
    if st.session_state.pop(LOGOUT_FLAG, False):
        # Sign-out: block stale cookie/URL; next reruns must not re-login.
        _qp_pop(SESSION_QP)
        _inject_clear_session_cookie()
        st.session_state[SESSION_RESTORE_RAN] = True
        return
    if st.session_state.get("user_id"):
        st.session_state[SESSION_RESTORE_RAN] = True
        return
    if st.session_state.get(SESSION_RESTORE_RAN):
        return
    st.session_state[SESSION_RESTORE_RAN] = True

    # 1) Server session in ?docubot_sid= (once per st.session_state; see SESSION_RESTORE_RAN)
    sid = _qp_get(SESSION_QP)
    if sid:
        uid = db.get_user_id_for_valid_session(sid)
        if uid:
            row = db.fetch_user_by_id(uid)
            if row and int(row.get("email_verified") or 0):
                _apply_user_row(row)
                return
        db.delete_user_session(sid)
        _qp_pop(SESSION_QP)
        # No st.rerun: fall through to cookie in the same run.

    # 2) Cookie (best effort)
    raw = read_session_cookie()
    if not raw:
        return
    parsed = verify_session_cookie_value(raw)
    if not parsed:
        _inject_clear_session_cookie()
        st.rerun()
        return
    pe_uid, em = parsed
    row = db.fetch_user_by_id(pe_uid)
    if not row or str(row.get("email", "")).lower() != em.lower():
        _inject_clear_session_cookie()
        st.rerun()
        return
    if not int(row.get("email_verified") or 0):
        _inject_clear_session_cookie()
        st.rerun()
        return
    _apply_user_row(row)
    # One-time: attach ?docubot_sid= so the next refresh does not depend on cookies
    try:
        nsid = secrets.token_urlsafe(32)
        db.insert_user_session(nsid, str(row["id"]), time.time() + _SESSION_TTL_SEC)
        st.query_params[SESSION_QP] = nsid
    except Exception:
        pass
    st.rerun()


def clear_all_session_persistence() -> None:
    """Sign out: remove server session, URL param, and cookie."""
    sid = _qp_get(SESSION_QP)
    if sid:
        db.delete_user_session(sid)
    _qp_pop(SESSION_QP)
    _inject_clear_session_cookie()
