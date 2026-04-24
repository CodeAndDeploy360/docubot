"""Register, login, email verification, password reset."""

from __future__ import annotations

import re
import secrets
import sqlite3
import time
import uuid

import bcrypt

from auth import db
from auth.mail import send_email
import config

_EMAIL_RE = re.compile(r"^[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}$")

_initialized = False

_VERIFICATION_TTL_SEC = 48 * 3600
_RESET_TTL_SEC = 2 * 3600


def init_auth() -> None:
    global _initialized
    if not _initialized:
        db.init_db()
        _initialized = True


def _validate_email(email: str) -> str | None:
    e = (email or "").strip().lower()
    if not e or not _EMAIL_RE.match(e):
        return "Enter a valid email address."
    return None


def _validate_password(password: str) -> str | None:
    if len(password) < 8:
        return "Password must be at least 8 characters."
    if len(password) > 256:
        return "Password is too long."
    return None


def _verification_url(token: str) -> str:
    return f"{config.public_base_url()}/?verify={token}"


def _reset_url(token: str) -> str:
    return f"{config.public_base_url()}/?reset={token}"


def register_user(email: str, password: str, password_confirm: str) -> tuple[str | None, str | None, str | None]:
    """
    Create account. Returns (user_id, error, in_app_message).
    in_app_message is set when email could not be sent (e.g. no SMTP) — show it to the user.
    """
    init_auth()
    if err := _validate_email(email):
        return None, err, None
    if err := _validate_password(password):
        return None, err, None
    if password != password_confirm:
        return None, "Passwords do not match.", None
    if db.fetch_user_by_email(email):
        return None, "An account with this email already exists.", None

    uid = str(uuid.uuid4())
    pw_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    now = time.time()
    v_token = secrets.token_urlsafe(32)
    v_exp = now + _VERIFICATION_TTL_SEC
    try:
        db.insert_user(
            uid,
            email,
            pw_hash,
            now,
            email_verified=0,
            verification_token=v_token,
            verification_expires=v_exp,
        )
    except sqlite3.IntegrityError:
        return None, "An account with this email already exists.", None
    except Exception as exc:  # noqa: BLE001
        return None, f"Could not create account: {exc}", None

    subj = "Verify your DocuBot account"
    body = f"Open this link to verify your email (expires in 48 hours):\n\n{_verification_url(v_token)}\n"
    if config.smtp_config():
        send_err = send_email(email, subj, body)
        if send_err:
            return None, f"Account created but email failed to send: {send_err}. Contact support or try again later.", None
        return None, None, "Check your **inbox** (and spam) for a verification link. You can sign in after you verify."
    return None, None, f"**Open this link** to verify your email, then sign in here:\n\n{_verification_url(v_token)}"


def verify_email_with_token(token: str) -> str | None:
    """Returns None on success, or an error string."""
    init_auth()
    t = (token or "").strip()
    if not t:
        return "Missing verification token."
    row = db.fetch_user_by_verification_token(t)
    if not row:
        return "This verification link is invalid or was already used."
    exp = row.get("verification_expires")
    if exp is not None and time.time() > float(exp):
        return "This link has expired. Register again or contact support."
    if int(row.get("email_verified") or 0):
        return None
    db.update_user_verification(
        str(row["id"]),
        email_verified=1,
        verification_token=None,
        verification_expires=None,
    )
    return None


def login_user(email: str, password: str) -> tuple[str | None, str | None, str | None]:
    """Returns (user_id, email, error)."""
    init_auth()
    e = (email or "").strip().lower()
    if not e:
        return None, None, "Enter your email and password."
    if err := _validate_email(e):
        return None, None, err
    row = db.fetch_user_by_email(e)
    if row is None:
        return None, None, "Wrong email or password."
    ok = bcrypt.checkpw(password.encode("utf-8"), row["password_hash"])
    if not ok:
        return None, None, "Wrong email or password."
    if not int(row.get("email_verified") or 0):
        return None, None, "Please verify your email first (check the link we sent) or use **Resend** on the sign-in tab."
    return str(row["id"]), str(row["email"]), None


def request_password_reset(email: str) -> str | None:
    """
    Always returns None (success message is generic for privacy). Sends email or stores token for in-app.
    """
    init_auth()
    e = (email or "").strip().lower()
    if not e:
        return "Enter an email address."
    row = db.fetch_user_by_email(e)
    if not row:
        return None
    r_token = secrets.token_urlsafe(32)
    r_exp = time.time() + _RESET_TTL_SEC
    db.set_reset_token(str(row["id"]), r_token, r_exp)
    if config.smtp_config():
        subj = "Reset your DocuBot password"
        body = f"Open this link to set a new password (expires in 2 hours):\n\n{_reset_url(r_token)}\n"
        send_err = send_email(e, subj, body)
        if send_err:
            return f"Could not send email: {send_err}"
    return None


def get_password_reset_link_for_email(email: str) -> str | None:
    """If SMTP is off, return a one-time link for display (same as request flow). For UI after request."""
    e = (email or "").strip().lower()
    row = db.fetch_user_by_email(e)
    if not row or not row.get("reset_token"):
        return None
    return _reset_url(str(row["reset_token"]))


def reset_password_with_token(token: str, new_password: str, confirm: str) -> str | None:
    """None on success."""
    if new_password != confirm:
        return "Passwords do not match."
    if err := _validate_password(new_password):
        return err
    row = db.fetch_user_by_reset_token((token or "").strip())
    if not row:
        return "This reset link is invalid or was already used."
    exp = row.get("reset_expires")
    if exp is not None and time.time() > float(exp):
        return "This link has expired. Request a new password reset."
    h = bcrypt.hashpw(new_password.encode("utf-8"), bcrypt.gensalt(rounds=12))
    db.update_password_and_clear_reset(str(row["id"]), h)
    return None


def resend_verification_email(email: str) -> tuple[str | None, str | None]:
    """
    Re-send verification for an unverified account. Returns (error, in_app_message_with_link).
    """
    init_auth()
    e = (email or "").strip().lower()
    row = db.fetch_user_by_email(e)
    if not row:
        return "If that email is registered, we sent instructions.", None
    if int(row.get("email_verified") or 0):
        return "This email is already verified.", None
    v_token = secrets.token_urlsafe(32)
    v_exp = time.time() + _VERIFICATION_TTL_SEC
    db.update_user_verification(
        str(row["id"]),
        email_verified=0,
        verification_token=v_token,
        verification_expires=v_exp,
    )
    subj = "Verify your DocuBot account"
    body = f"Open this link to verify your email:\n\n{_verification_url(v_token)}\n"
    if config.smtp_config():
        send_err = send_email(e, subj, body)
        if send_err:
            return f"Could not send: {send_err}", None
        return None, "We sent a new verification link to your inbox."
    return None, f"**Use this link to verify:**\n\n{_verification_url(v_token)}"
