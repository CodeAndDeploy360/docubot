"""SQLite user store (email-based, verification + password-reset tokens)."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from config import data_dir


def db_path() -> Path:
    return data_dir() / "users.sqlite3"


def get_connection() -> sqlite3.Connection:
    data_dir().mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path()), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create SQLite tables for email-based accounts and optional server sessions. No legacy migrations."""
    conn = get_connection()
    # Use IF NOT EXISTS: Streamlit (and multi-worker runs) can call init concurrently; a plain
    # "check then create" pair races and raises "table X already exist".
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id TEXT PRIMARY KEY,
            email TEXT NOT NULL UNIQUE COLLATE NOCASE,
            password_hash BLOB NOT NULL,
            email_verified INTEGER NOT NULL DEFAULT 0,
            verification_token TEXT,
            verification_expires REAL,
            reset_token TEXT,
            reset_expires REAL,
            created_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS user_sessions (
            id TEXT PRIMARY KEY,
            user_id TEXT NOT NULL,
            expires_at REAL NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()


def insert_user_session(session_id: str, user_id: str, expires_at: float) -> None:
    init_db()
    with get_connection() as c:
        c.execute(
            "INSERT INTO user_sessions (id, user_id, expires_at) VALUES (?, ?, ?)",
            (session_id, user_id, expires_at),
        )
        c.commit()


def get_user_id_for_valid_session(session_id: str) -> str | None:
    """Return user_id if session exists and is not expired; else None. Drops expired row."""
    if not (session_id or "").strip():
        return None
    init_db()
    now = time.time()
    sid = str(session_id).strip()
    with get_connection() as c:
        c.execute("DELETE FROM user_sessions WHERE expires_at < ?", (now,))
        cur = c.execute(
            "SELECT user_id FROM user_sessions WHERE id = ? AND expires_at > ?",
            (sid, now),
        )
        row = cur.fetchone()
        c.commit()
    if row is None:
        return None
    return str(row[0])


def delete_user_session(session_id: str) -> None:
    if not (session_id or "").strip():
        return
    init_db()
    with get_connection() as c:
        c.execute("DELETE FROM user_sessions WHERE id = ?", (str(session_id).strip(),))
        c.commit()


def delete_user_sessions_for_user(user_id: str) -> None:
    init_db()
    with get_connection() as c:
        c.execute("DELETE FROM user_sessions WHERE user_id = ?", (user_id,))
        c.commit()


def fetch_user_by_email(email: str) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        cur = conn.execute(
            "SELECT * FROM users WHERE email = ? COLLATE NOCASE",
            (email.strip().lower(),),
        )
        row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def fetch_user_by_id(user_id: str) -> dict[str, Any] | None:
    init_db()
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
        row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def fetch_user_by_reset_token(token: str) -> dict[str, Any] | None:
    if not token or not str(token).strip():
        return None
    init_db()
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM users WHERE reset_token = ?", (token.strip(),))
        row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def fetch_user_by_verification_token(token: str) -> dict[str, Any] | None:
    if not token or not str(token).strip():
        return None
    init_db()
    with get_connection() as conn:
        cur = conn.execute("SELECT * FROM users WHERE verification_token = ?", (token.strip(),))
        row = cur.fetchone()
    if row is None:
        return None
    return dict(row)


def insert_user(
    user_id: str,
    email: str,
    password_hash: bytes,
    created_at: float,
    *,
    email_verified: int = 0,
    verification_token: str | None = None,
    verification_expires: float | None = None,
) -> None:
    init_db()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO users (id, email, password_hash, email_verified, verification_token, verification_expires, reset_token, reset_expires, created_at)
            VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, ?)
            """,
            (user_id, email.strip().lower(), password_hash, email_verified, verification_token, verification_expires, created_at),
        )
        conn.commit()


def update_user_verification(
    user_id: str,
    *,
    email_verified: int,
    verification_token: str | None = None,
    verification_expires: float | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            UPDATE users SET email_verified = ?, verification_token = ?, verification_expires = ?
            WHERE id = ?
            """,
            (email_verified, verification_token, verification_expires, user_id),
        )
        conn.commit()


def set_reset_token(user_id: str, token: str | None, expires: float | None) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET reset_token = ?, reset_expires = ? WHERE id = ?",
            (token, expires, user_id),
        )
        conn.commit()


def update_password_and_clear_reset(user_id: str, new_password_hash: bytes) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE users SET password_hash = ?, reset_token = NULL, reset_expires = NULL WHERE id = ?",
            (new_password_hash, user_id),
        )
        conn.commit()
