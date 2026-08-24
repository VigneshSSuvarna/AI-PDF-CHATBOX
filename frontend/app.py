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
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# # ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ========================================================
       BLACK + CREAM PREMIUM THEME
       ======================================================== */

    :root {
        --black: #111111;
        --black-soft: #191919;

        --cream: #f4efe5;
        --cream-light: #faf7f0;
        --white: #ffffff;

        --text: #171717;
        --text-soft: #45413b;
        --muted: #746f67;

        --border: #d8d0c2;
        --border-dark: #292929;
    }


    /* ========================================================
       MAIN PAGE
       ======================================================== */

    .stApp {
        background: var(--cream) !important;
        color: var(--text) !important;
    }

    .main {
        background: var(--cream) !important;
    }


    /* ========================================================
       SIDEBAR
       ======================================================== */

    section[data-testid="stSidebar"] {
        background: var(--black) !important;
        border-right: 1px solid #252525 !important;
    }

    section[data-testid="stSidebar"] > div {
        background: var(--black) !important;
    }

    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #ddd8cf !important;
    }


    /* ========================================================
       BACKEND URL
       ======================================================== */

    section[data-testid="stSidebar"] input {
        background: #ffffff !important;
        color: #111111 !important;
        border: 1px solid #d2cabd !important;
        border-radius: 7px !important;
    }

    section[data-testid="stSidebar"] input::placeholder {
        color: #777777 !important;
    }


    /* ========================================================
       UPLOAD DOCUMENT
       ======================================================== */

    [data-testid="stFileUploader"] {
        background: var(--cream) !important;
        border: 1px solid #cfc6b8 !important;
        border-radius: 10px !important;
        padding: 4px !important;
    }

    [data-testid="stFileUploaderDropzone"] {
        background: var(--cream-light) !important;
        border: 1px dashed #bdb3a4 !important;
        border-radius: 8px !important;
    }

    /* Upload title / description */

    [data-testid="stFileUploader"] label {
        color: #171717 !important;
    }

    [data-testid="stFileUploader"] span {
        color: #292929 !important;
    }

    [data-testid="stFileUploader"] small {
        color: #625d55 !important;
    }

    /* Upload button */

    [data-testid="stFileUploader"] button {
        background: #ffffff !important;
        color: #171717 !important;
        border: 1px solid #c9c0b2 !important;
        border-radius: 8px !important;
    }

    [data-testid="stFileUploader"] button:hover {
        background: #eee8dd !important;
        color: #000000 !important;
        border-color: #aaa093 !important;
    }


    /* ========================================================
       DOCUMENT STATUS
       ======================================================== */

    /* Streamlit info box */

    [data-testid="stAlert"] {
        background: var(--cream) !important;
        border: 1px solid #cfc6b8 !important;
        border-radius: 10px !important;
        color: #171717 !important;
    }

    [data-testid="stAlert"] p {
        color: #292929 !important;
    }

    [data-testid="stAlert"] div {
        color: #292929 !important;
    }

    [data-testid="stAlert"] span {
        color: #292929 !important;
    }


    /* ========================================================
       STATUS BOX
       ======================================================== */

    .status-box {
        background: #1a1a1a !important;
        border: 1px solid #303030 !important;
        color: #eeeeee !important;
        border-radius: 8px !important;
    }

    .status-box * {
        color: #eeeeee !important;
    }


    /* ========================================================
       DOCUMENT / SOURCE BOX
       ======================================================== */

    .source-box {
        background: #1a1a1a !important;
        border: 1px solid #303030 !important;
        color: #eeeeee !important;
        border-radius: 8px !important;
    }

    .source-box * {
        color: #eeeeee !important;
    }

    .small-text {
        color: #a6a19a !important;
    }


    /* ========================================================
       MAIN HEADINGS
       ======================================================== */

    h1,
    h2,
    h3,
    h4 {
        color: #111111 !important;
    }

    .main-title {
        color: #111111 !important;
    }

    .subtitle {
        color: #625d55 !important;
    }


    /* ========================================================
       KNOWLEDGE MESSAGE
       ======================================================== */

    .scope-banner {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        color: #222222 !important;
        border-radius: 8px !important;
    }

    .scope-banner * {
        color: #222222 !important;
    }


    /* ========================================================
       CHAT SEARCH BAR
       ======================================================== */

    [data-testid="stChatInput"] {
        background: #ffffff !important;
        border: 1px solid #cfc7bb !important;
        border-radius: 10px !important;
        box-shadow: 0 3px 12px rgba(0, 0, 0, 0.06) !important;
    }

    [data-testid="stChatInput"] textarea {
        background: #ffffff !important;
        color: #111111 !important;
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #777777 !important;
    }


    /* ========================================================
       BUTTONS
       ======================================================== */

    .stButton > button {
        background: #111111 !important;
        color: #ffffff !important;
        border: 1px solid #111111 !important;
        border-radius: 7px !important;
    }

    .stButton > button:hover {
        background: #292929 !important;
        color: #ffffff !important;
    }


    /* ========================================================
       CHAT MESSAGES
       ======================================================== */

    [data-testid="stChatMessage"] {
        background: #ffffff !important;
        border: 1px solid #ddd6ca !important;
        border-radius: 10px !important;
    }

    [data-testid="stChatMessage"] p {
        color: #222222 !important;
    }


    /* ========================================================
       EXPANDERS
       ======================================================== */

    [data-testid="stExpander"] {
        background: #ffffff !important;
        border: 1px solid #d8d0c3 !important;
        border-radius: 8px !important;
    }


    /* ========================================================
       DIVIDERS
       ======================================================== */

    hr {
        border-color: #d4ccbf !important;
    }


    /* ========================================================
       SCROLLBAR
       ======================================================== */

    ::-webkit-scrollbar {
        width: 7px;
    }

    ::-webkit-scrollbar-track {
        background: var(--cream);
    }

    ::-webkit-scrollbar-thumb {
        background: #aaa297;
        border-radius: 10px;
    }

    ::-webkit-scrollbar-thumb:hover {
        background: #777067;
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
    # Name of currently active/indexed document
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
    Send a question to FastAPI /chat and yield
    streamed response chunks.
    """

    payload = {
        "session_id": session_id,
        "message": question,
    }

    # --------------------------------------------------------
    # Add document ID when a document is selected
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
                "Accept": "text/event-stream"
            },
        ) as response:

            # ------------------------------------------------
            # HTTP ERROR
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
                    f"Backend returned HTTP "
                    f"{response.status_code}: {detail}"
                )

            # ------------------------------------------------
            # READ SSE STREAM
            # ------------------------------------------------

            for raw_line in response.iter_lines(
                decode_unicode=True
            ):

                if raw_line is None:
                    continue

                line = raw_line.rstrip("\r")

                if not line:
                    continue

                if not line.startswith("data:"):

                    continue

                data = line[len("data:"):]

                if data.startswith(" "):

                    data = data[1:]

                # ------------------------------------------------
                # END OF STREAM
                # ------------------------------------------------

                if data == "[DONE]":

                    break

                if not data:

                    continue

                # ------------------------------------------------
                # PARSE JSON
                # ------------------------------------------------

                try:

                    parsed = json.loads(data)

                    if isinstance(parsed, dict):

                        # ----------------------------------------
                        # Normal streamed token
                        # ----------------------------------------

                        if "token" in parsed:

                            yield str(
                                parsed["token"]
                            )

                        # ----------------------------------------
                        # Alternative content format
                        # ----------------------------------------

                        elif "content" in parsed:

                            yield str(
                                parsed["content"]
                            )

                        # ----------------------------------------
                        # Alternative text format
                        # ----------------------------------------

                        elif "text" in parsed:

                            yield str(
                                parsed["text"]
                            )

                        # ----------------------------------------
                        # Backend error event
                        # ----------------------------------------

                        elif (
                            parsed.get("type")
                            == "error"
                        ):

                            error_text = parsed.get(
                                "error",
                                parsed.get(
                                    "token",
                                    "LLM generation failed.",
                                ),
                            )

                            raise RuntimeError(
                                str(error_text)
                            )

                        else:

                            continue

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

        # Safely get role and content
        role = message.get(
            "role",
            "assistant",
        )

        content = message.get(
            "content",
            "",
        )

        if role == "user":

            with st.chat_message("user"):

                st.markdown(content)

        else:

            with st.chat_message("assistant"):

                st.markdown(content)

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
    Display source information if the backend provides it.
    """

    if not sources:

        return

    with st.expander(
        "🤖 Sources Used",
        expanded=False,
    ):

        for index, source in enumerate(
            sources,
            start=1,
        ):

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
                            "Similarity: "
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

    # --------------------------------------------------------
    # SAVE USER MESSAGE
    # --------------------------------------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    # --------------------------------------------------------
    # DISPLAY USER MESSAGE
    # --------------------------------------------------------

    with st.chat_message("user"):

        st.markdown(question)

    # --------------------------------------------------------
    # DISPLAY ASSISTANT RESPONSE
    # --------------------------------------------------------

    with st.chat_message("assistant"):

        response_placeholder = st.empty()

        full_response = ""

        try:

            # ------------------------------------------------
            # STREAM RESPONSE
            # ------------------------------------------------

            for chunk in stream_chat_response(
                question=question,
                session_id=(
                    st.session_state.session_id
                ),
                doc_id=(
                    st.session_state.doc_id
                ),
            ):

                full_response += chunk

                response_placeholder.markdown(
                    full_response
                )

            # ------------------------------------------------
            # EMPTY RESPONSE
            # ------------------------------------------------

            if not full_response.strip():

                full_response = (
                    "The AI did not return a response."
                )

                response_placeholder.warning(
                    full_response
                )

            # ------------------------------------------------
            # SAVE RESPONSE
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
        # BACKEND URL
        # ----------------------------------------------------

        api_url = st.text_input(
            "Backend URL",
            value=st.session_state.api_url,
            help="URL where FastAPI is running.",
        )

        if api_url:

            st.session_state.api_url = (
                api_url.strip().rstrip("/")
            )

        # ----------------------------------------------------
        # BACKEND STATUS
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
        # CONVERSATION
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

        st.divider()

        # ----------------------------------------------------
        # UPLOAD DOCUMENT
        # ----------------------------------------------------

        st.subheader(
            "📄 Upload Document"
        )

        uploaded_file = st.file_uploader(
            "Drop your PDF here",
            type=["pdf"],
        )

        if uploaded_file is not None:

            if st.button(
                "Upload to System",
                use_container_width=True,
            ):

                with st.spinner(
                    "Processing PDF into Database..."
                ):

                    try:

                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                "application/pdf",
                            )
                        }

                        response = requests.post(
                            f"{get_api_url()}"
                            f"{UPLOAD_ENDPOINT}",
                            files=files,
                            timeout=REQUEST_TIMEOUT,
                        )

                        # ------------------------------------------------
                        # READ RESPONSE
                        # ------------------------------------------------

                        try:

                            data = response.json()

                        except ValueError:

                            data = {}

                        # ------------------------------------------------
                        # UPLOAD ERROR
                        # ------------------------------------------------

                        if response.status_code != 200:

                            detail = data.get(
                                "detail",
                                response.text,
                            )

                            st.error(
                                f"❌ Upload failed: {detail}"
                            )

                        else:

                            upload_status = data.get(
                                "status"
                            )

                            # ------------------------------------------------
                            # IMPORTANT:
                            #
                            # Backend UploadResponse is:
                            #
                            # {
                            #     "doc_id": "...",
                            #     "filename": "...",
                            #     "status": "indexed",
                            #     "uploaded_at": "...",
                            #     "chunks": 10
                            # }
                            #
                            # Therefore use "chunks", NOT
                            # "chunks_indexed".
                            # ------------------------------------------------

                            chunks_indexed = data.get(
                                "chunks",
                                0,
                            )

                            # ------------------------------------------------
                            # ONLY SELECT DOCUMENT IF INDEXED
                            # ------------------------------------------------

                            if (
                                upload_status
                                == "indexed"
                                and data.get("doc_id")
                            ):

                                st.session_state.doc_id = (
                                    data["doc_id"]
                                )

                                st.session_state.doc_name = (
                                    uploaded_file.name
                                )

                                st.success(
                                    f"✅ "
                                    f"{uploaded_file.name} "
                                    f"indexed successfully "
                                    f"({chunks_indexed} chunks)."
                                )

                            else:

                                # ------------------------------------------------
                                # Do NOT keep invalid doc_id
                                # ------------------------------------------------

                                st.session_state.doc_id = None

                                st.session_state.doc_name = None

                                st.warning(
                                    f"⚠️ "
                                    f"{uploaded_file.name} "
                                    f"was uploaded but not indexed "
                                    f"(status: {upload_status}). "
                                    f"Chat will continue using "
                                    f"the general knowledge base."
                                )

                    except requests.exceptions.RequestException as error:

                        st.error(
                            f"❌ Failed to upload: {error}"
                        )

        st.divider()

        # ====================================================
        # DOCUMENT STATUS
        # ====================================================

        st.subheader(
            "Document Status"
        )

        if st.session_state.doc_id:

            st.success(
                "📄 Document selected"
            )

            if st.session_state.doc_name:

                st.caption(
                    st.session_state.doc_name
                )

            st.code(
                st.session_state.doc_id,
                language="text",
            )

            # ------------------------------------------------
            # UPDATED BEHAVIOR
            #
            # PDF is prioritized, but backend also retrieves
            # knowledge-base context.
            # ------------------------------------------------

            st.info(
                "📄 This PDF is available as additional context. "
                "The AI can also use general knowledge."
            )

            if st.button(
                "❌ Clear Document Filter",
                use_container_width=True,
            ):

                st.session_state.doc_id = None

                st.session_state.doc_name = None

                st.rerun()

        else:

            st.info(
                "🌐 General AI knowledge is available. "
                "Upload a PDF anytime to add document context."
            )

        st.divider()

        # ----------------------------------------------------
        # CLEAR CONVERSATION
        # ----------------------------------------------------

        if st.button(
            "🗑️ Clear Conversation",
            use_container_width=True,
        ):

            st.session_state.messages = []

            st.session_state.session_id = (
                str(uuid.uuid4())
            )

            st.session_state.doc_id = None

            st.session_state.doc_name = None

            st.rerun()

        st.divider()

        st.markdown(
            """
            ### 🤖 AI PDF Chatbox

            Ask questions about your indexed PDF
            documents using RAG.

            **Pipeline**

            PDF → ChromaDB → Retriever → Memory → LLM → Answer
            """
        )


# ============================================================
# CHAT SCOPE BANNER
# ============================================================

def render_scope_banner() -> None:
    """
    Show the current retrieval scope.

    When a PDF is selected:
        PDF context is prioritized.
        Knowledge-base context is also available.

    When no PDF is selected:
        The full knowledge base is searched.
    """

    if st.session_state.doc_id:

        doc_label = (
            st.session_state.doc_name
            or "the selected document"
        )

        st.markdown(
            f'<div class="scope-banner scope-banner-doc">'
            f'📄 <b>{doc_label}</b> is available as additional context. '
            f'The AI can use information from the PDF together with '
            f'general AI knowledge.'
            f'</div>',
            unsafe_allow_html=True,
        )

    else:

        st.markdown(
            '<div class="scope-banner scope-banner-global">'
            '🌐 General AI knowledge is available. '
            'Upload a PDF anytime to add document context.'
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
        '<div class="main-title">'
        '🤖 AI PDF Chatbox'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="subtitle">'
        'Ask questions about your PDF documents using '
        'Retrieval-Augmented Generation.'
        '</div>',
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # EMPTY STATE
    # --------------------------------------------------------

    if not st.session_state.messages:

        st.info(
            "👋 Welcome! Ask a question about "
            "your PDF or general knowledge below."
        )

        st.markdown(
            "### 🔮 Try asking"
        )

        col1, col2, col3 = st.columns(3)

        with col1:

            st.markdown(
                "**📖 Understanding**\n\n"
                "What is the main topic of the document?"
            )

        with col2:

            st.markdown(
                "**🔍 Explanation**\n\n"
                "Explain the key concepts in the document."
            )

        with col3:

            st.markdown(
                "**📝 Summary**\n\n"
                "Summarize the important points."
            )

    # --------------------------------------------------------
    # CHAT HISTORY
    # --------------------------------------------------------

    display_chat_history()

    # --------------------------------------------------------
    # CURRENT SCOPE
    # --------------------------------------------------------

    render_scope_banner()

    # --------------------------------------------------------
    # CHAT INPUT
    # --------------------------------------------------------

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