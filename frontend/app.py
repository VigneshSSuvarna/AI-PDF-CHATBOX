"""
frontend/app.py
===============

Week 4 - Member 1
AI PDF CHATBOX - Streamlit Chat Interface

Responsibilities:
    - Display the chatbot UI
    - Maintain frontend conversation state
    - Send questions to FastAPI /chat
    - Display streamed responses
    - Handle API errors
    - Maintain session_id
    - Allow conversation reset
    - Handle PDF Document Uploads

Backend:
    FastAPI running at:
        http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import uuid
from typing import Optional

import requests
import streamlit as st


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_API_URL = "http://127.0.0.1:8000"

CHAT_ENDPOINT = "/chat"
HEALTH_ENDPOINT = "/health"
UPLOAD_ENDPOINT = "/upload"

REQUEST_TIMEOUT = 120


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI PDF Chatbox",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    /* Main title */
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #666;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    /* Status box */
    .status-box {
        padding: 0.7rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }

    /* Source box */
    .source-box {
        padding: 0.8rem;
        border-radius: 0.5rem;
        border: 1px solid #ddd;
        margin-top: 0.5rem;
    }

    /* Small text */
    .small-text {
        font-size: 0.8rem;
        color: #777;
    }

    /* Scope banner shown above the chat input */
    .scope-banner {
        padding: 0.6rem 0.9rem;
        border-radius: 0.5rem;
        margin-bottom: 0.75rem;
        font-size: 0.9rem;
    }

    .scope-banner-doc {
        background-color: rgba(46, 160, 67, 0.15);
        border: 1px solid rgba(46, 160, 67, 0.4);
        color: #3fb950;
    }

    .scope-banner-global {
        background-color: rgba(88, 166, 255, 0.12);
        border: 1px solid rgba(88, 166, 255, 0.35);
        color: #58a6ff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE INITIALIZATION
# ============================================================

def initialize_session_state() -> None:
    """
    Initialize all Streamlit session variables.
    """
    # --------------------------------------------------------
    # Unique conversation ID
    # --------------------------------------------------------
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    # --------------------------------------------------------
    # Chat messages
    # --------------------------------------------------------
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --------------------------------------------------------
    # Current document ID
    # --------------------------------------------------------
    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None

    # --------------------------------------------------------
    # Name of the currently active/indexed document
    # --------------------------------------------------------
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None

    # --------------------------------------------------------
    # API URL
    # --------------------------------------------------------
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL

initialize_session_state()


# ============================================================
# HELPER: API URL
# ============================================================

def get_api_url() -> str:
    """
    Return normalized backend URL.
    """
    url = st.session_state.api_url.strip()
    return url.rstrip("/")


# ============================================================
# API HEALTH CHECK
# ============================================================

def check_backend() -> bool:
    """
    Check whether FastAPI backend is running.
    """
    try:
        response = requests.get(
            f"{get_api_url()}{HEALTH_ENDPOINT}",
            timeout=5,
        )
        return response.status_code == 200
    except requests.RequestException:
        return False


# ============================================================
# CHAT API STREAM
# ============================================================

def stream_chat_response(
    question: str,
    session_id: str,
    doc_id: Optional[str] = None,
):
    """
    Send a question to FastAPI /chat and yield streamed response chunks.
    """
    payload = {
        "session_id": session_id,
        "message": question,
    }

    # Add document ID only if available
    if doc_id:
        payload["doc_id"] = doc_id

    try:
        with requests.post(
            f"{get_api_url()}{CHAT_ENDPOINT}",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            headers={"Accept": "text/event-stream"},
        ) as response:

            # HTTP error
            if response.status_code != 200:
                try:
                    error_data = response.json()
                    detail = error_data.get("detail", "Unknown backend error.")
                except Exception:
                    detail = response.text
                raise RuntimeError(f"Backend returned HTTP {response.status_code}: {detail}")

            # ------------------------------------------------------
            # Read streaming response (SSE)
            #
            # IMPORTANT: `data` is scoped fresh on every iteration
            # and we `continue` immediately for any line that is
            # not a "data:" line. Previously, `data` was declared
            # only inside the `if line.startswith("data:")` branch,
            # which meant a non-"data:" line (e.g. a bare line from
            # a multi-line payload) would silently reuse the PREVIOUS
            # iteration's `data` value and reprocess it — causing
            # the same chunk to be yielded multiple times and
            # producing repeated/garbled text in the UI.
            # ------------------------------------------------------
            for raw_line in response.iter_lines(decode_unicode=True):
                if raw_line is None:
                    continue

                line = raw_line.rstrip("\r")

                if not line:
                    continue

                if not line.startswith("data:"):
                    # Not an SSE data line (e.g. blank/comment/other
                    # field) — skip it instead of reusing stale data.
                    continue

                data = line[len("data:"):]

                if data.startswith(" "):
                    data = data[1:]

                if data == "[DONE]":
                    break

                if not data:
                    continue

                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        if "token" in parsed:
                            yield str(parsed["token"])
                        elif "content" in parsed:
                            yield str(parsed["content"])
                        elif "text" in parsed:
                            yield str(parsed["text"])
                        else:
                            yield data
                    else:
                        yield str(parsed)
                except json.JSONDecodeError:
                    yield data

    except requests.exceptions.Timeout:
        raise RuntimeError("The backend request timed out. Please try again.")
    except requests.exceptions.ConnectionError:
        raise RuntimeError("Could not connect to the FastAPI backend. Make sure api_framework.py is running.")
    except requests.RequestException as error:
        raise RuntimeError(f"API request failed: {error}")


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

def display_chat_history() -> None:
    """
    Display all messages stored in Streamlit session state.
    """
    for message in st.session_state.messages:
        
        # Safely get role and content to prevent KeyErrors
        role = message.get("role", "assistant")
        content = message.get("content", "")

        if role == "user":
            with st.chat_message("user"):
                st.markdown(content)
        else:
            with st.chat_message("assistant"):
                st.markdown(content)

                sources = message.get("sources", [])
                if sources:
                    display_sources(sources)


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(sources: list) -> None:
    """
    Display source information if the backend provides it.
    """
    if not sources:
        return

    with st.expander("📚 Sources Used", expanded=False):
        for index, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                document = source.get("source", source.get("document", source.get("doc_id", "Unknown document")))
                page = source.get("page", source.get("page_number", None))
                score = source.get("score", source.get("similarity", None))
                text = source.get("text", "")

                st.markdown(f"**{index}. 📄 {document}**")

                if page is not None:
                    st.caption(f"Page: {page}")

                if score is not None:
                    try:
                        st.caption(f"Similarity: {float(score):.3f}")
                    except (ValueError, TypeError):
                        pass

                if text:
                    st.caption(text[:300] + ("..." if len(text) > 300 else ""))
            else:
                st.markdown(f"**{index}. 📄 {source}**")


# ============================================================
# SEND MESSAGE
# ============================================================

def process_user_message(question: str) -> None:
    """
    Send a question to the backend and display the streamed response.
    """
    question = question.strip()
    if not question:
        return

    # SAVE USER MESSAGE
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    # DISPLAY USER MESSAGE
    with st.chat_message("user"):
        st.markdown(question)

    # DISPLAY ASSISTANT RESPONSE
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Stream response
            for chunk in stream_chat_response(
                question=question,
                session_id=st.session_state.session_id,
                doc_id=st.session_state.doc_id,
            ):
                full_response += chunk
                response_placeholder.markdown(full_response)

            if not full_response.strip():
                full_response = "The AI did not return a response."
                response_placeholder.warning(full_response)

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": [],
                }
            )

        except RuntimeError as error:
            error_message = f"⚠️ {str(error)}"
            response_placeholder.error(error_message)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )

        except Exception as error:
            error_message = "⚠️ An unexpected error occurred."
            response_placeholder.error(error_message)
            print(f"Frontend error: {error}")
            
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )


# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    """
    Render application sidebar.
    """
    with st.sidebar:
        st.header("⚙️ Settings")

        # Backend URL
        api_url = st.text_input(
            "Backend URL",
            value=st.session_state.api_url,
            help="URL where FastAPI is running.",
        )

        if api_url:
            st.session_state.api_url = api_url.strip().rstrip("/")

        # Backend status
        if check_backend():
            st.success("🟢 Backend connected")
        else:
            st.error("🔴 Backend unavailable")

        st.divider()

        st.subheader("Conversation")
        st.caption("Session ID")
        st.code(st.session_state.session_id, language="text")

        st.divider()

        # Upload Document
        st.subheader("📄 Upload Document")
        uploaded_file = st.file_uploader("Drop your PDF here", type=["pdf"])

        if uploaded_file is not None:
            if st.button("Upload to System", use_container_width=True):
                with st.spinner("Processing PDF into Database..."):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        }

                        response = requests.post(
                            f"{get_api_url()}{UPLOAD_ENDPOINT}",
                            files=files,
                            timeout=REQUEST_TIMEOUT,
                        )

                        # --------------------------------------------------
                        # Don't blindly raise_for_status() before reading
                        # the body — ingestion failures come back as JSON
                        # with a useful "detail" message we want to show.
                        # --------------------------------------------------
                        try:
                            data = response.json()
                        except ValueError:
                            data = {}

                        if response.status_code != 200:
                            detail = data.get("detail", response.text)
                            st.error(f"❌ Upload failed: {detail}")

                        else:
                            upload_status = data.get("status")
                            chunks_indexed = data.get("chunks_indexed", 0)

                            # ------------------------------------------------
                            # CRITICAL: only attach doc_id to future chat
                            # requests if the backend actually indexed the
                            # document. Otherwise /chat would apply a
                            # metadata_filter={"doc_id": ...} that matches
                            # zero chunks, and every question would come
                            # back as "I do not know based on the provided
                            # context" — even ones the base knowledge base
                            # could easily answer.
                            # ------------------------------------------------
                            if upload_status == "indexed" and data.get("doc_id"):
                                st.session_state.doc_id = data["doc_id"]
                                st.session_state.doc_name = uploaded_file.name

                                st.success(
                                    f"✅ {uploaded_file.name} indexed "
                                    f"successfully ({chunks_indexed} chunks)."
                                )

                            else:
                                # Ingestion did not succeed — do NOT scope
                                # future chats to this doc_id.
                                st.session_state.doc_id = None
                                st.session_state.doc_name = None

                                st.warning(
                                    f"⚠️ {uploaded_file.name} was uploaded but "
                                    f"not indexed (status: {upload_status}). "
                                    "Chat will fall back to the general "
                                    "knowledge base."
                                )

                    except requests.exceptions.RequestException as error:
                        st.error(f"❌ Failed to upload: {error}")

        st.divider()

        # Document Status
        st.subheader("Document Status")
        if st.session_state.doc_id:
            st.success("Document selected")
            if st.session_state.doc_name:
                st.caption(st.session_state.doc_name)
            st.code(st.session_state.doc_id, language="text")

            st.warning(
                "⚠️ Chat answers are currently scoped to ONLY this "
                "document. General knowledge-base questions (e.g. "
                "topics not covered in this PDF) will return "
                "'I do not know'."
            )

            if st.button("❌ Clear Document Filter", use_container_width=True):
                st.session_state.doc_id = None
                st.session_state.doc_name = None
                st.rerun()

        else:
            st.info(
                "No document selected.\n\n"
                "The chatbot will search the entire knowledge base."
            )

        st.divider()

        if st.button("🗑️ Clear Conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.doc_id = None
            st.session_state.doc_name = None
            st.rerun()

        st.divider()

        st.markdown(
            """
            ### 📚 AI PDF Chatbox
            Ask questions about your indexed PDF documents using RAG.

            **Pipeline**
            PDF → ChromaDB → Retriever → Memory → LLM → Answer
            """
        )


# ============================================================
# CHAT SCOPE BANNER
# ============================================================

def render_scope_banner() -> None:
    """
    Show a persistent, hard-to-miss banner above the chat input
    indicating whether answers are scoped to a single uploaded
    document or searching the entire knowledge base.

    This exists because attaching a doc_id filters ChromaDB to
    ONLY that document's chunks — if the question isn't answered
    within that specific PDF, the LLM will correctly say
    "I do not know", which can look like a bug if the scoping
    isn't visible to the user.
    """

    if st.session_state.doc_id:

        doc_label = st.session_state.doc_name or "the selected document"

        st.markdown(
            f'<div class="scope-banner scope-banner-doc">'
            f'🔍 Answers are scoped to <b>{doc_label}</b> only. '
            f'Use "Clear Document Filter" in the sidebar to search '
            f'the full knowledge base instead.'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="scope-banner scope-banner-global">'
            '🌐 Searching the full knowledge base '
            '(no document filter applied).'
            '</div>',
            unsafe_allow_html=True,
        )


# ============================================================
# MAIN UI
# ============================================================

def main() -> None:
    """
    Main Streamlit application.
    """
    render_sidebar()

    st.markdown(
        '<div class="main-title">📚 AI PDF Chatbox</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions about your PDF documents using Retrieval-Augmented Generation.'
        '</div>',
        unsafe_allow_html=True,
    )

    # Empty State
    if not st.session_state.messages:
        st.info("👋 Welcome! Ask a question about your PDF documents below.")
        st.markdown("### 💡 Try asking")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("**📖 Understanding**\n\nWhat is the main topic of the document?")
        with col2:
            st.markdown("**🔍 Explanation**\n\nExplain the key concepts in the document.")
        with col3:
            st.markdown("**📝 Summary**\n\nSummarize the important points.")

    # Chat History & Input
    display_chat_history()

    render_scope_banner()

    question = st.chat_input("Ask a question about your PDF...")
    if question:
        process_user_message(question)

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()