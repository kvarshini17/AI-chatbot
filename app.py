"""
app.py
======
Main Streamlit entrypoint for the AI Assistant. Wires together the sidebar,
chat UI, chat history, Ollama client, and PDF-RAG engine. Run with:

    streamlit run app.py
"""

from __future__ import annotations

import streamlit as st  # type: ignore

from components import sidebar, chat_ui
from core.chat import send_user_message, finalize_assistant_reply
from core.history import HistoryManager
from core.ollama_client import OllamaClient, OllamaConnectionError
from core.pdf_chat import PDFChatEngine
from core.typing_animation import render_thinking_indicator, stream_with_typing_effect
from utils.file_utils import load_settings

st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ------------------------------------------------------------------------
# Cached / singleton resources
# ------------------------------------------------------------------------
@st.cache_resource
def get_history_manager() -> HistoryManager:
    return HistoryManager()


@st.cache_resource
def get_ollama_client() -> OllamaClient:
    return OllamaClient()


@st.cache_resource
def get_pdf_engine() -> PDFChatEngine:
    return PDFChatEngine()


def init_session_state() -> None:
    """Initialize all session_state keys used across the app."""
    defaults = {
        "current_conv_id": None,
        "pdf_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def main() -> None:
    init_session_state()
    settings = load_settings()
    chat_ui.inject_custom_css(font_size=settings["font_size"])

    history = get_history_manager()
    ollama_client = get_ollama_client()
    pdf_engine = get_pdf_engine()

    sidebar.render_sidebar(history, ollama_client, pdf_engine)

    # Resolve (or lazily create) the active conversation.
    conv = None
    if st.session_state.current_conv_id:
        conv = history.get_conversation(st.session_state.current_conv_id)

    if conv is None:
        conv = history.create_conversation(model=settings["model"])
        st.session_state.current_conv_id = conv.id

    mode_label = "📄 PDF Chat" if st.session_state.pdf_mode else "💬 Chat"
    st.markdown(f"#### {mode_label} · *{conv.title}*")

    if not conv.messages:
        chat_ui.render_empty_state()
    else:
        chat_ui.render_conversation(conv)

    user_input = st.chat_input("Message the assistant…")
    if user_input:
        _handle_user_turn(user_input, conv, history, ollama_client, pdf_engine, settings)


def _handle_user_turn(
    user_input: str,
    conv,
    history: HistoryManager,
    ollama_client: OllamaClient,
    pdf_engine: PDFChatEngine,
    settings: dict,
) -> None:
    """Handle a single round-trip: render the user's message, stream the
    assistant's reply with a typing effect, and persist everything."""
    chat_ui.render_message("user", user_input)

    placeholder = st.empty()
    render_thinking_indicator(placeholder)

    try:
        if st.session_state.pdf_mode:
            generator, sources = pdf_engine.answer_question(
                query=user_input,
                model=conv.model,
                client=ollama_client,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                max_tokens=settings["max_tokens"],
            )
            conv.messages.append({"role": "user", "content": user_input})
            full_text = stream_with_typing_effect(
                placeholder, generator, settings["typing_speed_ms"]
            )
            finalize_assistant_reply(conv, full_text, history)
            chat_ui.render_sources(sources)
        else:
            generator = send_user_message(
                conv, user_input, ollama_client, history,
                temperature=settings["temperature"],
                top_p=settings["top_p"],
                max_tokens=settings["max_tokens"],
            )
            full_text = stream_with_typing_effect(
                placeholder, generator, settings["typing_speed_ms"]
            )
            finalize_assistant_reply(conv, full_text, history)

    except OllamaConnectionError as exc:
        placeholder.error(f"⚠️ {exc}")
    except Exception as exc:  # noqa: BLE001 - never let the app crash
        placeholder.error(f"⚠️ Something went wrong: {exc}")

    st.rerun()


if __name__ == "__main__":
    main()
