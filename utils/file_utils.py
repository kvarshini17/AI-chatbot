"""
file_utils.py
=============
Small, reusable helper functions for file handling, formatting, and
miscellaneous tasks used throughout the app. Keeping these separate avoids
duplicate code in the UI and core modules.
"""

from __future__ import annotations

import json
import re
import unicodedata
from datetime import datetime
from pathlib import Path
from typing import Any

from utils.constants import SETTINGS_FILE, DEFAULT_TEMPERATURE, DEFAULT_MAX_TOKENS, \
    DEFAULT_TOP_P, DEFAULT_MODEL, DEFAULT_TYPING_SPEED_MS, DEFAULT_FONT_SIZE


def sanitize_filename(name: str, max_length: int = 60) -> str:
    """
    Convert an arbitrary string into a filesystem-safe filename.

    Args:
        name: Raw string (e.g. a chat title) to sanitize.
        max_length: Maximum number of characters to keep.

    Returns:
        A sanitized, lowercase, hyphen-separated filename fragment.
    """
    if not name:
        return "untitled"
    normalized = unicodedata.normalize("NFKD", name)
    normalized = normalized.encode("ascii", "ignore").decode("ascii")
    normalized = re.sub(r"[^\w\s-]", "", normalized).strip().lower()
    normalized = re.sub(r"[-\s]+", "-", normalized)
    return normalized[:max_length] or "untitled"


def timestamped_filename(base_name: str, extension: str) -> str:
    """
    Build a filename that includes the current date and time, as required
    for exports (e.g. "project-update_2026-07-14_14-32-05.txt").

    Args:
        base_name: Human readable base (usually the chat title).
        extension: File extension without the leading dot.

    Returns:
        A complete, sanitized filename.
    """
    safe_base = sanitize_filename(base_name)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{safe_base}_{stamp}.{extension}"


def human_readable_size(num_bytes: int) -> str:
    """Convert a byte count into a human readable string (KB/MB/GB)."""
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def get_directory_size(path: Path) -> int:
    """
    Recursively compute the total size (in bytes) of everything under a
    directory. Used to display storage usage in the sidebar.
    """
    total = 0
    if not path.exists():
        return 0
    for entry in path.rglob("*"):
        if entry.is_file():
            try:
                total += entry.stat().st_size
            except OSError:
                # File may have been deleted mid-scan; ignore gracefully.
                continue
    return total


def load_settings() -> dict[str, Any]:
    """
    Load persisted user settings from disk. Returns sensible defaults if the
    settings file doesn't exist yet or is corrupted.
    """
    defaults = {
        "model": DEFAULT_MODEL,
        "temperature": DEFAULT_TEMPERATURE,
        "max_tokens": DEFAULT_MAX_TOKENS,
        "top_p": DEFAULT_TOP_P,
        "theme": "Dark",
        "typing_speed_ms": DEFAULT_TYPING_SPEED_MS,
        "font_size": DEFAULT_FONT_SIZE,
    }
    if not SETTINGS_FILE.exists():
        return defaults
    try:
        with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
            saved = json.load(f)
        defaults.update(saved)
        return defaults
    except (json.JSONDecodeError, OSError):
        # Corrupted settings file -- fall back to defaults rather than crash.
        return defaults


def save_settings(settings: dict[str, Any]) -> None:
    """Persist user settings to disk as JSON."""
    try:
        with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
            json.dump(settings, f, indent=2)
    except OSError as exc:
        raise RuntimeError(f"Could not save settings: {exc}") from exc


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to a maximum length, appending an ellipsis if cut."""
    text = text.strip().replace("\n", " ")
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def format_relative_date(dt: datetime) -> str:
    """Return a friendly relative-date label for a given datetime."""
    now = datetime.now()
    delta_days = (now.date() - dt.date()).days
    if delta_days == 0:
        return "Today"
    if delta_days == 1:
        return "Yesterday"
    if delta_days <= 7:
        return "Last 7 Days"
    return "Older"
