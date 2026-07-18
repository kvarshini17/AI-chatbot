"""
history.py
==========
SQLite persistence layer for chat conversations. Handles creating,
reading, updating, deleting, searching and grouping conversations so the
rest of the app never writes raw SQL.
"""

from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Iterator

from utils.constants import DATABASE_PATH
from utils.file_utils import format_relative_date, truncate_text


@dataclass
class Conversation:
    """In-memory representation of a single chat conversation."""

    id: str
    title: str
    model: str
    created_at: str
    updated_at: str
    messages: list[dict] = field(default_factory=list)


class HistoryError(Exception):
    """Raised for any database-related failure in the history layer."""


class HistoryManager:
    """CRUD interface over the SQLite chat-history database."""

    def __init__(self, db_path=DATABASE_PATH) -> None:
        self.db_path = str(db_path)
        self._init_db()

    # ------------------------------------------------------------------
    # Connection handling
    # ------------------------------------------------------------------
    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Context manager yielding a connection with sane defaults."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except sqlite3.OperationalError as exc:
            conn.rollback()
            if "locked" in str(exc).lower():
                raise HistoryError(
                    "The database is currently busy. Please try again."
                ) from exc
            raise HistoryError(f"Database error: {exc}") from exc
        except sqlite3.Error as exc:
            conn.rollback()
            raise HistoryError(f"Database error: {exc}") from exc
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create the conversations table if it doesn't already exist."""
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    messages TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_updated_at "
                "ON conversations(updated_at DESC)"
            )

    # ------------------------------------------------------------------
    # Create / update
    # ------------------------------------------------------------------
    def create_conversation(self, model: str, first_message: str = "") -> Conversation:
        """Create and persist a brand-new, empty conversation."""
        now = datetime.now().isoformat()
        conv = Conversation(
            id=str(uuid.uuid4()),
            title=self._generate_title(first_message),
            model=model,
            created_at=now,
            updated_at=now,
            messages=[],
        )
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO conversations (id, title, model, created_at, "
                "updated_at, messages) VALUES (?, ?, ?, ?, ?, ?)",
                (conv.id, conv.title, conv.model, conv.created_at,
                 conv.updated_at, json.dumps(conv.messages)),
            )
        return conv

    def save_conversation(self, conv: Conversation) -> None:
        """Persist an updated conversation (messages, title, model)."""
        conv.updated_at = datetime.now().isoformat()
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title=?, model=?, updated_at=?, "
                "messages=? WHERE id=?",
                (conv.title, conv.model, conv.updated_at,
                 json.dumps(conv.messages), conv.id),
            )

    def rename_conversation(self, conv_id: str, new_title: str) -> None:
        """Rename a conversation."""
        new_title = new_title.strip() or "Untitled Chat"
        with self._connect() as conn:
            conn.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE id=?",
                (new_title, datetime.now().isoformat(), conv_id),
            )

    def delete_conversation(self, conv_id: str) -> None:
        """Permanently delete a conversation."""
        with self._connect() as conn:
            conn.execute("DELETE FROM conversations WHERE id=?", (conv_id,))

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------
    def get_conversation(self, conv_id: str) -> Conversation | None:
        """Fetch a single conversation by id, or None if not found."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM conversations WHERE id=?", (conv_id,)
            ).fetchone()
        if row is None:
            return None
        return self._row_to_conversation(row)

    def list_conversations(self) -> list[Conversation]:
        """Return all conversations, most recently updated first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations ORDER BY updated_at DESC"
            ).fetchall()
        return [self._row_to_conversation(r) for r in rows]

    def search_conversations(self, query: str) -> list[Conversation]:
        """Search conversation titles and message content for a query string."""
        query = query.strip().lower()
        if not query:
            return self.list_conversations()
        matches = []
        for conv in self.list_conversations():
            if query in conv.title.lower():
                matches.append(conv)
                continue
            for msg in conv.messages:
                if query in msg.get("content", "").lower():
                    matches.append(conv)
                    break
        return matches

    def group_by_date(
        self, conversations: list[Conversation]
    ) -> dict[str, list[Conversation]]:
        """Group conversations into Today / Yesterday / Last 7 Days / Older."""
        groups: dict[str, list[Conversation]] = {
            "Today": [], "Yesterday": [], "Last 7 Days": [], "Older": [],
        }
        for conv in conversations:
            try:
                dt = datetime.fromisoformat(conv.updated_at)
            except ValueError:
                dt = datetime.now()
            groups[format_relative_date(dt)].append(conv)
        return groups

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _row_to_conversation(row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            title=row["title"],
            model=row["model"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            messages=json.loads(row["messages"]),
        )

    @staticmethod
    def _generate_title(first_message: str) -> str:
        """Auto-generate a conversation title from the first user message."""
        if not first_message.strip():
            return "New Chat"
        return truncate_text(first_message, max_length=45)
