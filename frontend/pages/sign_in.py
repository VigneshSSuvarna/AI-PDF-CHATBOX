"""
frontend/pages/sign_in.py
"""

import streamlit as st

from auth_db import (
    email_exists,
    username_exists,
    create_user,
    authenticate_user,
)


st.set_page_config(
    page_title="Sign In - AI Chatbox",
    page_icon="🔐",
    layout="centered",
)


# =========================================================
# STYLING
# =========================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

    .stApp {
        background-color: #0d1117 !important;
        color: #e6edf3 !important;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }

    header {
        display: none !important;
    }

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

    .stTextInput > div > div > input {
        background-color: #0d1117 !important;
        color: white !important;
        border: 1px solid #30363d !important;
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


# =========================================================
# INITIALIZE SESSION VARIABLES
# =========================================================

if "username_authenticated" not in st.session_state:
    st.session_state.username_authenticated = False

if "show_account_setup" not in st.session_state:
    st.session_state.show_account_setup = False


# =========================================================
# GOOGLE LOGIN
# DO NOT CHANGE THIS
# =========================================================

if not st.user.is_logged_in:

    if st.button("⬅️ Back to Chat", use_container_width=False):
        st.switch_page("app.py")

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

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        # YOUR EXISTING GOOGLE LOGIN
        if st.button("Sign in with Google", type="primary"):
            st.login()

        st.caption(
            "You will enter your Gmail password securely on Google's sign-in page."
        )

    st.stop()


# =========================================================
# GOOGLE USER INFORMATION
# =========================================================

google_email = (
    getattr(st.user, "email", None)
    or getattr(st.user, "preferred_username", None)
)

google_name = (
    getattr(st.user, "name", None)
    or getattr(st.user, "given_name", None)
    or "Google User"
)


if not google_email:
    st.error("Unable to retrieve your Google email address.")
    st.stop()


# =========================================================
# CHECK IF THIS GOOGLE ACCOUNT ALREADY HAS A USERNAME
# =========================================================

has_account = email_exists(google_email)


# =========================================================
# NEW USER → CREATE USERNAME + PASSWORD
# =========================================================

if not has_account:

    st.markdown(
        """
        <div class="auth-card">
            <div class="auth-title">Create your account</div>
            <div class="auth-subtitle">
                Your Google account has been verified
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:

        st.info(f"Verified Google account: {google_email}")

        username = st.text_input(
            "Username",
            placeholder="Choose a unique username",
        )

        password = st.text_input(
            "Password",
            type="password",
            placeholder="Create your password",
        )

        confirm_password = st.text_input(
            "Confirm Password",
            type="password",
            placeholder="Enter your password again",
        )

        if st.button("Create Account", type="primary"):

            username = username.strip()

            if not username:
                st.error("Please enter a username.")

            elif not password:
                st.error("Please enter a password.")

            elif len(username) < 3:
                st.error("Username must contain at least 3 characters.")

            elif password != confirm_password:
                st.error("Passwords do not match.")

            elif username_exists(username):
                st.error("That username is already taken. Choose another.")

            else:

                success, message = create_user(
                    username,
                    google_email,
                    password,
                )

                if success:

                    st.session_state.user_name = username
                    st.session_state.user_email = google_email
                    st.session_state.user_role = "User"
                    st.session_state.is_authenticated = True
                    st.session_state.username_authenticated = True
                    st.session_state.access_token = None

                    st.success("Account created successfully!")

                    st.switch_page("app.py")

                else:
                    st.error(message)

    st.stop()


# =========================================================
# EXISTING USER → USERNAME + PASSWORD LOGIN
# =========================================================

st.markdown(
    """
    <div class="auth-card">
        <div class="auth-title">Welcome back</div>
        <div class="auth-subtitle">
            Enter your username and password
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


col1, col2, col3 = st.columns([1, 2, 1])

with col2:

    st.info(f"Google account verified: {google_email}")

    username = st.text_input(
        "Username",
        placeholder="Enter your username",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter your password",
    )

    if st.button("Sign In", type="primary"):

        if not username:
            st.error("Please enter your username.")

        elif not password:
            st.error("Please enter your password.")

        else:

            valid = authenticate_user(
                username,
                password,
                google_email,
            )

            if valid:

                st.session_state.user_name = username
                st.session_state.user_email = google_email
                st.session_state.user_role = "User"
                st.session_state.is_authenticated = True
                st.session_state.username_authenticated = True
                st.session_state.access_token = None

                st.success("Login successful!")

                st.switch_page("app.py")

            else:
                st.error(
                    "Invalid username or password for this Google account."
                )