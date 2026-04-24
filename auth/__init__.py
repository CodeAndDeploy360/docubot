"""User accounts (SQLite) and session wiring for per-user Chroma storage."""

from auth.service import (
    get_password_reset_link_for_email,
    init_auth,
    login_user,
    register_user,
    request_password_reset,
    resend_verification_email,
    reset_password_with_token,
    verify_email_with_token,
)

__all__ = [
    "get_password_reset_link_for_email",
    "init_auth",
    "login_user",
    "register_user",
    "request_password_reset",
    "resend_verification_email",
    "reset_password_with_token",
    "verify_email_with_token",
]
