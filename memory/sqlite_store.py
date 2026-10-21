"""
memory/sqlite_store.py — SQLite wrapper for persisting research session history.

Responsibilities:
  - Save a completed research session (query + report + metadata)
  - Retrieve past sessions for display in the UI
  - Provide a simple audit trail of what was researched

sqlite3 is part of Python's standard library — no installation needed.
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Optional

from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)


class SQLiteStore:
    """
    Manages a simple sessions table in a local SQLite database.

    Schema:
        sessions (
            id            INTEGER  PRIMARY KEY AUTOINCREMENT,
            query         TEXT     NOT NULL,   -- original user query
            final_report  TEXT,                -- markdown report from Writer
            iterations    INTEGER  DEFAULT 0,  -- how many Critic loops ran
            findings_count INTEGER DEFAULT 0,  -- how many pages were scraped
            created_at    TEXT     NOT NULL    -- ISO timestamp
        )

    Usage:
        db = SQLiteStore()
        session_id = db.save_session(query="...", report="...", ...)
        history    = db.get_recent_sessions(limit=5)
        session    = db.get_session_by_id(session_id)
    """

    def __init__(self):
        """
        Open (or create) the SQLite database and ensure the table exists.
        Called once when the app starts.
        """
        logger.info(f"Opening SQLite database: {SQLITE_DB_PATH}")
        # check_same_thread=False is needed when FastAPI uses threads
        self._conn = sqlite3.connect(
            SQLITE_DB_PATH,
            check_same_thread=False,
        )
        # Return rows as dict-like objects instead of plain tuples
        self._conn.row_factory = sqlite3.Row
        self._create_table()
        logger.info("SQLiteStore ready")

    def _create_table(self) -> None:
        """
        Create the sessions table if it doesn't already exist.
        Safe to call on every startup — IF NOT EXISTS prevents errors.
        """
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                query          TEXT    NOT NULL,
                final_report   TEXT,
                iterations     INTEGER DEFAULT 0,
                findings_count INTEGER DEFAULT 0,
                created_at     TEXT    NOT NULL
            )
        """)
        self._conn.commit()

    def save_session(
        self,
        query: str,
        final_report: Optional[str],
        iterations: int = 0,
        findings_count: int = 0,
    ) -> int:
        """
        Persist a completed research session.

        Args:
            query:          The original user research query.
            final_report:   Markdown report string from the Writer agent.
            iterations:     Number of Critic→Researcher loops that ran.
            findings_count: Total number of scraped findings stored.

        Returns:
            The new session's integer ID (useful for linking to UI state).
            Returns -1 on failure.
        """
        try:
            cursor = self._conn.execute(
                """
                INSERT INTO sessions
                    (query, final_report, iterations, findings_count, created_at)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                (
                    query,
                    final_report,
                    iterations,
                    findings_count,
                    datetime.utcnow().isoformat(),
                ),
            )
            self._conn.commit()
            session_id = cursor.lastrowid
            logger.info(f"Saved session #{session_id}: '{query[:50]}'")
            return session_id

        except Exception as e:
            logger.error(f"Failed to save session: {e}")
            return -1

    def get_recent_sessions(self, limit: int = 10) -> List[Dict]:
        """
        Retrieve the most recent research sessions.

        Args:
            limit: Maximum number of sessions to return.

        Returns:
            List of dicts ordered newest-first, each with keys:
            id, query, final_report, iterations, findings_count, created_at.
            Returns empty list on failure.
        """
        try:
            cursor = self._conn.execute(
                """
                SELECT id, query, final_report, iterations,
                       findings_count, created_at
                FROM   sessions
                ORDER  BY id DESC
                LIMIT  ?
                """,
                (limit,),
            )
            # sqlite3.Row supports dict(row) conversion
            rows = [dict(row) for row in cursor.fetchall()]
            logger.info(f"Retrieved {len(rows)} recent sessions")
            return rows

        except Exception as e:
            logger.error(f"Failed to get sessions: {e}")
            return []

    def get_session_by_id(self, session_id: int) -> Optional[Dict]:
        """
        Retrieve a single session by its ID.

        Args:
            session_id: The integer ID from save_session().

        Returns:
            Dict with all session fields, or None if not found.
        """
        try:
            cursor = self._conn.execute(
                "SELECT * FROM sessions WHERE id = ?",
                (session_id,),
            )
            row = cursor.fetchone()
            return dict(row) if row else None

        except Exception as e:
            logger.error(f"Failed to get session {session_id}: {e}")
            return None

    def close(self) -> None:
        """Close the database connection cleanly."""
        self._conn.close()
        logger.info("SQLiteStore connection closed")