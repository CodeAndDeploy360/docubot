"""Send transactional email via SMTP (optional; callers use in-app links when SMTP is unset)."""

from __future__ import annotations

import smtplib
from email.message import EmailMessage
from typing import Any

import config


def send_email(to_addr: str, subject: str, body_text: str) -> str | None:
    """
    Send one message via configured SMTP. Returns None on success, or an error string on failure.
    If SMTP is not configured, returns an error and does not send; callers should use in-app links instead
    and normally avoid calling this when :func:`config.smtp_config` is None.
    """
    cfg: dict[str, Any] | None = config.smtp_config()
    if not cfg:
        return "SMTP is not configured"

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = cfg["from_addr"]
    msg["To"] = to_addr
    msg.set_content(body_text)

    host = cfg["host"]
    port = int(cfg.get("port", 587))
    user = cfg.get("user")
    password = cfg.get("password")

    try:
        if port == 465:
            with smtplib.SMTP_SSL(host, port, timeout=30) as s:
                if user and password is not None:
                    s.login(user, password or "")
                s.send_message(msg)
        else:
            with smtplib.SMTP(host, port, timeout=30) as s:
                if cfg.get("use_tls", True):
                    s.starttls()
                if user and password is not None:
                    s.login(user, password or "")
                s.send_message(msg)
    except OSError as exc:  # noqa: BLE001
        return str(exc)
    return None
