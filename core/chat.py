"""
chat.py
=======
Core orchestration for a normal (non-PDF) chat turn: builds the message
list sent to Ollama, streams the response, and keeps the Conversation
object in sync with the database.
"""

from __future__ import annotations

from typing import Generator

from core.history import Conversation, HistoryManager
from core.ollama_client import OllamaClient, OllamaConnectionError


def build_message_payload(conv: Conversation) -> list[dict[str, str]]:
    """Convert a Conversation's stored messages into the Ollama chat format."""
    return [
        {"role": m["role"], "content": m["content"]}
        for m in conv.messages
        if m["role"] in ("user", "assistant", "system")
    ]


def send_user_message(
    conv: Conversation,
    user_text: str,
    client: OllamaClient,
    history: HistoryManager,
    temperature: float,
    top_p: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    """
    Append the user's message to the conversation, then stream back the
    assistant's reply. The caller is responsible for rendering the stream
    (e.g. via typing_animation) and must call `finalize_assistant_reply`
    once the full text has been collected.

    Raises:
        OllamaConnectionError: propagated from the client so the UI layer
            can display a friendly error message.
    """
    conv.messages.append({"role": "user", "content": user_text})
    history.save_conversation(conv)

    payload = build_message_payload(conv)
    yield from client.stream_chat(
        model=conv.model,
        messages=payload,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )


def finalize_assistant_reply(
    conv: Conversation, assistant_text: str, history: HistoryManager
) -> None:
    """Persist the assistant's finished reply to the conversation."""
    if not assistant_text.strip():
        assistant_text = "_(The model returned an empty response.)_"
    conv.messages.append({"role": "assistant", "content": assistant_text})
    history.save_conversation(conv)
