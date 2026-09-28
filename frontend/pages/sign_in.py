"""
frontend/pages/sign_in.py
"""

import streamlit as st

st.set_page_config(
    page_title="Sign In - AI Chatbox",
    page_icon="🔐",
    layout="centered",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    header { display: none !important; }

    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    .auth-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 16px;
        padding: 40px;
        max-width: 450px;
        margin: 40px auto 10px auto;
        text-align: center;
        box-shadow: 0 8px 24px rgba(0,0,0,0.2);
    }

    .auth-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f0f6fc;
    }

    .auth-subtitle {
        color: #8b949e;
        margin-bottom: 2rem;
    }

    .stButton > button {
        width: 100% !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    [data-testid="baseButton-primary"] {
        background: #58a6ff !important;
        color: #0d1117 !important;
        margin-top: 15px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Check whether Google user is already logged in
# ---------------------------------------------------------

if hasattr(st.user, "is_logged_in") and st.user.is_logged_in:

    # Store Google account information in Streamlit session
    st.session_state.user_name = (
        getattr(st.user, "name", None)
        or getattr(st.user, "given_name", None)
        or "Google User"
    )

    st.session_state.user_email = (
        getattr(st.user, "email", None)
        or getattr(st.user, "preferred_username", None)
    )

    st.session_state.user_role = "User"
    st.session_state.is_authenticated = True

    # Google OAuth gives us identity, not a JWT created by our backend.
    # Keep this empty until backend authentication is added.
    st.session_state.access_token = None

    st.switch_page("app.py")


# ---------------------------------------------------------
# Back to Chat
# ---------------------------------------------------------

if st.button("⬅️ Back to Chat", use_container_width=False):
    st.switch_page("app.py")


# ---------------------------------------------------------
# Login Card
# ---------------------------------------------------------

st.markdown(
    """
    <div class="auth-card">
        <div class="auth-title">Welcome back</div>
        <div class="auth-subtitle">
            Sign in securely with your Google account
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Google Login
# ---------------------------------------------------------

col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    if st.button("Sign in with Google", type="primary"):
        st.login()

    st.caption(
        "You will enter your Gmail password securely on Google's sign-in page."
    )