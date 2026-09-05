"""
frontend/app.py
===============

Week 4 - Member 1
AI PDF CHATBOX - Modern Streamlit Interface

Responsibilities:
    - Display modern, responsive chatbot UI
    - Maintain frontend conversation state & session tracking
    - Stream responses in real-time from FastAPI /chat
    - Handle PDF uploads and display indexing status
    - Provide interactive quick-prompt starters
    - Display structured source citations
    - PERSISTENT CHAT HISTORY (ChatGPT-style sidebar)
    - Dynamic User Profile & Clean Collapsible Settings
    - Multi-page Auth Routing
"""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Generator, Optional

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

# Directory to save chat history locally
CHATS_DIR = Path("frontend/chats")
CHATS_DIR.mkdir(parents=True, exist_ok=True)

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
# MODERN SAAS CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    /* Main Container & Canvas */
    .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
    }

    /* HIDE STREAMLIT HEADER, DEPLOY BUTTON, AND TOOLBAR */
    header {display: none !important;}
    footer {visibility: hidden !important;}
    [data-testid="stSidebarNav"] {display: none !important;}
    .stAppDeployButton {display: none !important;}
    [data-testid="stToolbar"] {display: none !important;}

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161b22 !important;
        border-right: 1px solid #30363d !important;
    }

    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #f0f6fc !important;
        font-weight: 600;
        font-size: 1.05rem;
    }

    /* User Profile Box */
    .user-profile {
        display: flex; align-items: center; gap: 12px; padding: 14px;
        background: #1c2128; border: 1px solid #30363d; border-radius: 12px; margin-bottom: 1.5rem;
    }

    .user-avatar {
        width: 42px; height: 42px; border-radius: 50%;
        background: linear-gradient(135deg, #58a6ff 0%, #a371f7 100%);
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 1.1rem; color: #ffffff; flex-shrink: 0;
    }

    .user-info { display: flex; flex-direction: column; overflow: hidden; }
    .user-name { font-weight: 600; font-size: 0.95rem; color: #f0f6fc; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
    .user-role { font-size: 0.75rem; color: #8b949e; }

    /* Hero Header */
    .hero-container { padding: 0.2rem 0 0.8rem 0; border-bottom: 1px solid #21262d; margin-bottom: 1.2rem; }
    .hero-title {
        font-size: 2.1rem; font-weight: 700; margin: 0;
        background: linear-gradient(90deg, #58a6ff 0%, #a371f7 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .hero-subtitle { color: #8b949e; font-size: 0.95rem; margin-top: 0.3rem; }

    /* Scope Pills */
    .scope-pill {
        display: inline-flex; align-items: center; gap: 8px; padding: 6px 14px;
        border-radius: 20px; font-size: 0.85rem; font-weight: 500; margin-bottom: 1.2rem;
    }
    .scope-pill-doc { background: rgba(56, 139, 253, 0.15); color: #58a6ff; border: 1px solid rgba(56, 139, 253, 0.35); }
    .scope-pill-global { background: rgba(63, 185, 80, 0.12); color: #3fb950; border: 1px solid rgba(63, 185, 80, 0.3); }

    /* Chat Messages */
    [data-testid="stChatMessage"] {
        background: #161b22 !important; border: 1px solid #30363d !important;
        border-radius: 12px !important; padding: 14px 18px !important; margin-bottom: 14px !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    [data-testid="stChatMessage"][data-testid*="user"] { background: #1f242c !important; border: 1px solid #3b434e !important; }

    /* Chat Input */
    [data-testid="stChatInput"] {
        background: #161b22 !important; border: 1px solid #30363d !important;
        border-radius: 12px !important; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3) !important;
    }
    [data-testid="stChatInput"] textarea { color: #f0f6fc !important; font-size: 0.95rem !important; }

    /* Standard Secondary Buttons */
    .stButton > button {
        background: #21262d !important; color: #f0f6fc !important; border: 1px solid #363b42 !important;
        border-radius: 8px !important; font-weight: 500 !important; transition: all 0.2s ease; text-align: left !important;
    }
    .stButton > button:hover { background: #30363d !important; border-color: #8b949e !important; color: #58a6ff !important; transform: translateY(-1px); }

    /* Primary Action Button (Sign Up) */
    [data-testid="baseButton-primary"] { background: #58a6ff !important; color: #0d1117 !important; border: none !important; text-align: center !important; }
    [data-testid="baseButton-primary"]:hover { background: #79c0ff !important; color: #0d1117 !important; }

    /* Sidebar Metric Cards */
    .metric-card { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px 14px; margin: 8px 0; }
    .metric-card-title { color: #8b949e; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card-value { color: #f0f6fc; font-size: 0.95rem; font-weight: 600; margin-top: 2px; word-break: break-all; }

    /* Code styling */
    code { font-family: 'JetBrains Mono', monospace !important; color: #79c0ff !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# STATE INITIALIZATION & HELPERS
# ============================================================

def get_initials(name: str) -> str:
    """Generate up to 2 capital initials from the user name."""
    parts = name.strip().split()
    if not parts:
        return "GU"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[-1][0]).upper()

def initialize_session_state() -> None:
    """Initialize state variables across Streamlit reruns."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "doc_id" not in st.session_state:
        st.session_state.doc_id = None
    if "doc_name" not in st.session_state:
        st.session_state.doc_name = None
    if "doc_chunks" not in st.session_state:
        st.session_state.doc_chunks = 0
    if "api_url" not in st.session_state:
        st.session_state.api_url = DEFAULT_API_URL
    if "pending_prompt" not in st.session_state:
        st.session_state.pending_prompt = None

    # Dynamic Auth Session Defaults
    if "user_name" not in st.session_state:
        st.session_state.user_name = "Guest User"
    if "user_role" not in st.session_state:
        st.session_state.user_role = "Guest"
    if "is_authenticated" not in st.session_state:
        st.session_state.is_authenticated = False

initialize_session_state()

# ============================================================
# CALLBACK FUNCTIONS (Instant UI resets)
# ============================================================

def start_new_chat() -> None:
    """Callback to cleanly reset session variables for a fresh chat."""
    st.session_state.messages = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.doc_id = None
    st.session_state.doc_name = None
    st.session_state.doc_chunks = 0
    st.session_state.pending_prompt = None

def delete_current_chat() -> None:
    """Callback to delete the current JSON file and start a new chat."""
    file_path = CHATS_DIR / f"{st.session_state.session_id}.json"
    if file_path.exists():
        try:
            file_path.unlink()
        except Exception:
            pass
    start_new_chat()

def load_historical_chat(sess_id: str, messages: list, doc_id: Optional[str], doc_name: Optional[str]) -> None:
    """Callback to instantly load a previously saved chat."""
    st.session_state.session_id = sess_id
    st.session_state.messages = messages
    st.session_state.doc_id = doc_id
    st.session_state.doc_name = doc_name

def remove_active_scope() -> None:
    """Callback to remove current PDF context."""
    st.session_state.doc_id = None
    st.session_state.doc_name = None
    st.session_state.doc_chunks = 0

def set_quick_prompt(prompt: str) -> None:
    """Callback to trigger a prompt immediately."""
    st.session_state.pending_prompt = prompt

def sign_out() -> None:
    """Callback to reset user auth."""
    st.session_state.user_name = "Guest User"
    st.session_state.user_role = "Guest"
    st.session_state.is_authenticated = False

# ============================================================
# DATA LOGIC
# ============================================================

def save_chat_session() -> None:
    """Saves the current session to a local JSON file for the sidebar history."""
    if not st.session_state.messages:
        return
        
    first_user_msg = next((m["content"] for m in st.session_state.messages if m["role"] == "user"), "New Chat")
    title = first_user_msg[:28] + "..." if len(first_user_msg) > 28 else first_user_msg

    chat_data = {
        "session_id": st.session_state.session_id,
        "title": title,
        "messages": st.session_state.messages,
        "doc_id": st.session_state.doc_id,
        "doc_name": st.session_state.doc_name,
    }

    file_path = CHATS_DIR / f"{st.session_state.session_id}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(chat_data, f, indent=4)

def get_api_url() -> str:
    return st.session_state.api_url.strip().rstrip("/")

def check_backend_status() -> tuple[bool, str]:
    try:
        res = requests.get(f"{get_api_url()}{HEALTH_ENDPOINT}", timeout=3)
        if res.status_code == 200:
            data = res.json()
            return True, f"Online ({data.get('retriever', 'ready')})"
        return False, "Error"
    except Exception:
        return False, "Offline"

def stream_chat_response(question: str, session_id: str, doc_id: Optional[str] = None) -> Generator[str, None, None]:
    payload = {"session_id": session_id, "message": question}
    if doc_id:
        payload["doc_id"] = doc_id

    try:
        with requests.post(f"{get_api_url()}{CHAT_ENDPOINT}", json=payload, stream=True, timeout=REQUEST_TIMEOUT) as response:
            if response.status_code != 200:
                raise RuntimeError(f"Backend HTTP {response.status_code}: {response.text}")

            for raw_line in response.iter_lines(decode_unicode=True):
                if not raw_line: continue
                line = raw_line.rstrip("\r")
                if not line.startswith("data:"): continue

                data = line[len("data:"):].strip()
                if data == "[DONE]": break
                if not data: continue

                try:
                    parsed = json.loads(data)
                    if isinstance(parsed, dict):
                        yield str(parsed.get("content", parsed.get("token", parsed.get("text", ""))))
                    else:
                        yield str(parsed)
                except json.JSONDecodeError:
                    yield data

    except Exception as e:
        raise RuntimeError(f"API Error: {e}")

def display_sources(sources: list) -> None:
    if not sources: return
    with st.expander("📚 Sources Referenced", expanded=False):
        for index, source in enumerate(sources, start=1):
            if isinstance(source, dict):
                document = source.get("source", source.get("document", source.get("doc_id", "Unknown Document")))
                page = source.get("page", source.get("page_number", None))
                st.markdown(f"**{index}. 📄 {document}**")
                if page is not None:
                    st.caption(f"📍 Page: {page}")
            else:
                st.markdown(f"**{index}. 📄 {source}**")

# ============================================================
# SIDEBAR
# ============================================================

def render_sidebar() -> None:
    with st.sidebar:
        
        # User Profile
        initials = get_initials(st.session_state.user_name)
        st.markdown(
            f"""
            <div class="user-profile">
                <div class="user-avatar">{initials}</div>
                <div class="user-info">
                    <span class="user-name">{st.session_state.user_name}</span>
                    <span class="user-role">{st.session_state.user_role}</span>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        # Document Ingestion
        st.markdown("### 📄 Document Ingestion")
        uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"], label_visibility="collapsed")
        
        if uploaded_file is not None:
            if st.button("🚀 Process & Index PDF", use_container_width=True):
                with st.spinner("Chunking & Embedding into ChromaDB..."):
                    try:
                        files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                        response = requests.post(f"{get_api_url()}{UPLOAD_ENDPOINT}", files=files, timeout=REQUEST_TIMEOUT)
                        if response.status_code == 200:
                            data = response.json()
                            st.session_state.doc_id = data.get("doc_id")
                            st.session_state.doc_name = uploaded_file.name
                            st.toast(f"✅ Ingested {uploaded_file.name} successfully!", icon="🎉")
                            st.rerun()
                        else:
                            st.error(f"Upload failed: {response.text}")
                    except Exception as err:
                        st.error(f"Upload connection error: {err}")

        # Active Scope
        if st.session_state.doc_id:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-card-title">Active PDF Scope</div>
                    <div class="metric-card-value">📄 {st.session_state.doc_name}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.button("❌ Remove Active Scope", on_click=remove_active_scope, use_container_width=True)

        st.markdown("---")
        
        # Recent Chats
        st.markdown("### 📝 Recent Chats")
        col1, col2 = st.columns(2)
        
        # Using on_click callbacks for instant response
        col1.button("➕ New Chat", on_click=start_new_chat, use_container_width=True)
        col2.button("🗑️ Delete", on_click=delete_current_chat, use_container_width=True)
            
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)
        
        chat_files = list(CHATS_DIR.glob("*.json"))
        if chat_files:
            chat_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            for file_path in chat_files:
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        
                    sess_id = data.get("session_id")
                    title = data.get("title", "Unknown Chat")
                    is_active = (sess_id == st.session_state.session_id)
                    btn_label = f"🟢 {title}" if is_active else f"💬 {title}"
                    
                    if st.button(btn_label, key=f"hist_{sess_id}", on_click=load_historical_chat, args=(sess_id, data.get("messages", []), data.get("doc_id"), data.get("doc_name")), use_container_width=True):
                        pass # Callback handles state update
                except Exception:
                    continue
        else:
            st.caption("No previous chats found.")

        st.markdown("---")

        # Settings
        with st.expander("⚙️ Settings & System Status", expanded=False):
            is_connected, status_text = check_backend_status()
            color = "rgba(63, 185, 80, 0.12)" if is_connected else "rgba(248,81,73,0.15)"
            text_color = "#3fb950" if is_connected else "#f85149"
            
            st.markdown(f'<div class="scope-pill scope-pill-global" style="margin-bottom: 10px; width:100%; background:{color}; color:{text_color};">{"🟢" if is_connected else "🔴"} {status_text}</div>', unsafe_allow_html=True)

            api_url = st.text_input("Backend Endpoint", value=st.session_state.api_url)
            if api_url:
                st.session_state.api_url = api_url.strip().rstrip("/")

# ============================================================
# CHAT LOGIC
# ============================================================

def process_message(prompt_text: str) -> None:
    prompt_text = prompt_text.strip()
    if not prompt_text:
        return

    st.session_state.messages.append({"role": "user", "content": prompt_text})
    save_chat_session()
    
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt_text)

    with st.chat_message("assistant", avatar="⚡"):
        placeholder = st.empty()
        full_response = ""

        try:
            for chunk in stream_chat_response(prompt_text, st.session_state.session_id, st.session_state.doc_id):
                full_response += chunk
                placeholder.markdown(full_response + "▌")

            placeholder.markdown(full_response or "I could not generate an answer.")
            st.session_state.messages.append({"role": "assistant", "content": full_response, "sources": []})
            save_chat_session()

        except Exception as error:
            error_msg = f"⚠️ **Error:** {str(error)}"
            placeholder.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg, "sources": []})
            save_chat_session()

# ============================================================
# MAIN APPLICATION
# ============================================================

def main() -> None:
    render_sidebar()

    # Authentication Top Right Buttons
    if st.session_state.is_authenticated:
        _, user_display_col, signout_col = st.columns([7.2, 1.6, 1.2])
        with user_display_col:
            st.markdown(f"<div style='text-align:right; padding-top:8px; color:#8b949e; font-size:0.9rem;'>Logged in as <b style='color:#f0f6fc;'>{st.session_state.user_name}</b></div>", unsafe_allow_html=True)
        with signout_col:
            st.button("Sign Out", on_click=sign_out, use_container_width=True)
    else:
        spacer_col, signin_col, signup_col = st.columns([7.5, 1.25, 1.25])
        with signin_col:
            if st.button("Sign In", use_container_width=True):
                st.switch_page("pages/sign_in.py")
        with signup_col:
            if st.button("Sign Up", type="primary", use_container_width=True):
                st.switch_page("pages/sign_up.py")
            
    st.markdown("<div style='margin-bottom: 0.5rem;'></div>", unsafe_allow_html=True)

    # Hero Banner
    st.markdown(
        """
        <div class="hero-container">
            <h1 class="hero-title">AI PDF Chatbox</h1>
            <div class="hero-subtitle">High-precision Retrieval-Augmented Generation (RAG) assistant for your documents</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Scope Pill
    if st.session_state.doc_id:
        st.markdown(f'<div class="scope-pill scope-pill-doc">🎯 Scoped to: <b>{st.session_state.doc_name}</b></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="scope-pill scope-pill-global">🌐 Searching full knowledge base (No document filter applied)</div>', unsafe_allow_html=True)

    # Empty State & Quick Starters
    if not st.session_state.messages:
        st.markdown("#### 💡 Quick Starters")
        c1, c2, c3 = st.columns(3)
        c1.button("📖 Document Summary\n\nSummarize key takeaways", on_click=set_quick_prompt, args=("Summarize the key takeaways and main concepts of the document.",), use_container_width=True)
        c2.button("🔍 Concept Breakdown\n\nExplain key technical details", on_click=set_quick_prompt, args=("Explain the core architecture and key technical concepts in the document.",), use_container_width=True)
        c3.button("📝 Action Items & Risks\n\nHighlight important items", on_click=set_quick_prompt, args=("List any limitations, challenges, or notable considerations described.",), use_container_width=True)
        st.markdown("---")

    # Render Active Chat
    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        avatar = "🧑‍💻" if role == "user" else "⚡"
        with st.chat_message(role, avatar=avatar):
            st.markdown(msg.get("content", ""))
            sources = msg.get("sources", [])
            if sources:
                display_sources(sources)

    # Execute pending quick prompt if one was clicked
    if st.session_state.pending_prompt:
        prompt_to_run = st.session_state.pending_prompt
        st.session_state.pending_prompt = None
        process_message(prompt_to_run)

    # Chat Input Box
    if user_input := st.chat_input("Ask a question about your uploaded PDF or general knowledge..."):
        process_message(user_input)

if __name__ == "__main__":
    main()