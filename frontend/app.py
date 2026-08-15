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

        st.session_state.session_id = str(
            uuid.uuid4()
        )

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

    # Remove trailing slash

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
    Send a question to FastAPI /chat and yield
    streamed response chunks.

    Expected backend request:

        {
            "session_id": "...",
            "message": "...",
            "doc_id": "..."
        }

    Expected backend response:

        data: token

        data: token

        data: [DONE]
    """

    payload = {
        "session_id": session_id,
        "message": question,
    }

    # --------------------------------------------------------
    # Add document ID only if available
    # --------------------------------------------------------

    if doc_id:

        payload["doc_id"] = doc_id

    try:

        with requests.post(
            f"{get_api_url()}{CHAT_ENDPOINT}",
            json=payload,
            stream=True,
            timeout=REQUEST_TIMEOUT,
            headers={
                "Accept": "text/event-stream",
            },
        ) as response:

            # ------------------------------------------------
            # HTTP error
            # ------------------------------------------------

            if response.status_code != 200:

                try:

                    error_data = response.json()

                    detail = error_data.get(
                        "detail",
                        "Unknown backend error.",
                    )

                except Exception:

                    detail = response.text

                raise RuntimeError(
                    f"Backend returned "
                    f"HTTP {response.status_code}: "
                    f"{detail}"
                )

            # ------------------------------------------------
            # Read streaming response
            # ------------------------------------------------

            for raw_line in response.iter_lines(
                decode_unicode=True
            ):

                if raw_line is None:
                    continue

    # Only remove the line ending.
    # DO NOT use .strip() because streamed
    # tokens may intentionally begin with spaces.
                line = raw_line.rstrip("\r")

                if not line:
                    continue

                if line.startswith("data:"):

        # Remove "data:" itself.
                    data = line[len("data:"):]

        # Remove exactly ONE SSE separator space.
        # Preserve any additional whitespace belonging
        # to the actual token.
                if data.startswith(" "):
                    data = data[1:]

                else:

                    data = line

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

        raise RuntimeError(
            "The backend request timed out. "
            "Please try again."
        )

    except requests.exceptions.ConnectionError:

        raise RuntimeError(
            "Could not connect to the FastAPI backend. "
            "Make sure api_framework.py is running."
        )

    except requests.RequestException as error:

        raise RuntimeError(
            f"API request failed: {error}"
        )


# ============================================================
# DISPLAY CHAT HISTORY
# ============================================================

def display_chat_history() -> None:
    """
    Display all messages stored in Streamlit session state.
    """

    for message in st.session_state.messages:

        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        # ----------------------------------------------------
        # User message
        # ----------------------------------------------------

        if role == "user":

            with st.chat_message(
                "user"
            ):

                st.markdown(content)

        # ----------------------------------------------------
        # Assistant message
        # ----------------------------------------------------

        else:

            with st.chat_message(
                "assistant"
            ):

                st.markdown(content)

                # --------------------------------------------
                # Sources
                # --------------------------------------------

                sources = message.get(
                    "sources",
                    [],
                )

                if sources:

                    display_sources(
                        sources
                    )


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(
    sources: list,
) -> None:
    """
    Display source information if the backend
    provides it.

    This is intentionally included now so Member 3
    can later connect the source citation data.
    """

    if not sources:

        return

    with st.expander(
        "📚 Sources Used",
        expanded=False,
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

            # --------------------------------------------
            # Dictionary source
            # --------------------------------------------

            if isinstance(
                source,
                dict,
            ):

                document = source.get(
                    "source",
                    source.get(
                        "document",
                        source.get(
                            "doc_id",
                            "Unknown document",
                        ),
                    ),
                )

                page = source.get(
                    "page",
                    source.get(
                        "page_number",
                        None,
                    ),
                )

                score = source.get(
                    "score",
                    source.get(
                        "similarity",
                        None,
                    ),
                )

                text = source.get(
                    "text",
                    "",
                )

                st.markdown(
                    f"**{index}. 📄 {document}**"
                )

                if page is not None:

                    st.caption(
                        f"Page: {page}"
                    )

                if score is not None:

                    try:

                        st.caption(
                            f"Similarity: "
                            f"{float(score):.3f}"
                        )

                    except (
                        ValueError,
                        TypeError,
                    ):

                        pass

                if text:

                    st.caption(
                        text[:300]
                        + (
                            "..."
                            if len(text) > 300
                            else ""
                        )
                    )

            # --------------------------------------------
            # String source
            # --------------------------------------------

            else:

                st.markdown(
                    f"**{index}. 📄 {source}**"
                )


# ============================================================
# SEND MESSAGE
# ============================================================

def process_user_message(
    question: str,
) -> None:
    """
    Send a question to the backend and display
    the streamed response.
    """

    question = question.strip()

    if not question:

        return

    # ========================================================
    # SAVE USER MESSAGE
    # ========================================================

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # ========================================================
    # DISPLAY USER MESSAGE
    # ========================================================

    with st.chat_message(
        "user"
    ):

        st.markdown(question)

    # ========================================================
    # DISPLAY ASSISTANT RESPONSE
    # ========================================================

    with st.chat_message(
        "assistant"
    ):

        response_placeholder = st.empty()

        full_response = ""

        try:

            # ------------------------------------------------
            # Stream response
            # ------------------------------------------------

            for chunk in stream_chat_response(
                question=question,
                session_id=st.session_state.session_id,
                doc_id=st.session_state.doc_id,
            ):

                full_response += chunk

                response_placeholder.markdown(
                    full_response
                )

            # ------------------------------------------------
            # No response
            # ------------------------------------------------

            if not full_response.strip():

                full_response = (
                    "The AI did not return a response."
                )

                response_placeholder.warning(
                    full_response
                )

            # ------------------------------------------------
            # Save assistant response
            # ------------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "sources": [],
                }
            )

        except RuntimeError as error:

            error_message = (
                f"⚠️ {str(error)}"
            )

            response_placeholder.error(
                error_message
            )

            # --------------------------------------------
            # Save error message
            # --------------------------------------------

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_message,
                    "sources": [],
                }
            )

        except Exception as error:

            error_message = (
                "⚠️ An unexpected error occurred."
            )

            response_placeholder.error(
                error_message
            )

            print(
                f"Frontend error: {error}"
            )

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

        # ----------------------------------------------------
        # Backend URL
        # ----------------------------------------------------

        api_url = st.text_input(
            "Backend URL",
            value=st.session_state.api_url,
            help=(
                "URL where FastAPI is running."
            ),
        )

        if api_url:

            st.session_state.api_url = (
                api_url.strip().rstrip("/")
            )

        # ----------------------------------------------------
        # Backend status
        # ----------------------------------------------------

        if check_backend():

            st.success(
                "🟢 Backend connected"
            )

        else:

            st.error(
                "🔴 Backend unavailable"
            )

        st.divider()

        # ----------------------------------------------------
        # Session ID
        # ----------------------------------------------------

        st.subheader(
            "Conversation"
        )

        st.caption(
            "Session ID"
        )

        st.code(
            st.session_state.session_id,
            language="text",
        )

        # ----------------------------------------------------
        # Document
        # ----------------------------------------------------

        st.subheader(
            "Document"
        )

        if st.session_state.doc_id:

            st.success(
                "Document selected"
            )

            st.code(
                st.session_state.doc_id,
                language="text",
            )

        else:

            st.info(
                "No document selected.\n\n"
                "The chatbot will use the default "
                "retrieval collection."
            )

        # ----------------------------------------------------
        # Clear chat
        # ----------------------------------------------------

        st.divider()

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.session_state.session_id = str(
                uuid.uuid4()
            )

            st.rerun()

        # ----------------------------------------------------
        # About
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            """
            ### 📚 AI PDF Chatbox

            Ask questions about your indexed
            PDF documents using RAG.

            **Pipeline**

            PDF → ChromaDB → Retriever →
            Memory → LLM → Answer
            """
        )


# ============================================================
# MAIN UI
# ============================================================

def main() -> None:
    """
    Main Streamlit application.
    """

    # ========================================================
    # SIDEBAR
    # ========================================================

    render_sidebar()

    # ========================================================
    # HEADER
    # ========================================================

    st.markdown(
        '<div class="main-title">'
        '📚 AI PDF Chatbox'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions about your PDF documents '
        'using Retrieval-Augmented Generation.'
        '</div>',
        unsafe_allow_html=True,
    )

    # ========================================================
    # EMPTY STATE
    # ========================================================

    if not st.session_state.messages:

        st.info(
            "👋 Welcome! Ask a question about "
            "your PDF documents below."
        )

        # ----------------------------------------------------
        # Suggested questions
        # ----------------------------------------------------

        st.markdown(
            "### 💡 Try asking"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                """
                **📖 Understanding**

                What is the main topic
                of the document?
                """
            )

        with col2:

            st.markdown(
                """
                **🔍 Explanation**

                Explain the key concepts
                in the document.
                """
            )

        with col3:

            st.markdown(
                """
                **📝 Summary**

                Summarize the important
                points.
                """
            )

    # ========================================================
    # CHAT HISTORY
    # ========================================================

    display_chat_history()

    # ========================================================
    # CHAT INPUT
    # ========================================================

    question = st.chat_input(
        "Ask a question about your PDF..."
    )

    if question:

        process_user_message(
            question
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()