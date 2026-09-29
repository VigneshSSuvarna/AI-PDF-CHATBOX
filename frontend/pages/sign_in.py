"""
frontend/pages/sign_in.py
AI PDF Chatbox - Authentication
"""

import streamlit as st

from auth_db import (
    email_exists,
    username_exists,
    create_user,
    authenticate_user,
)


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="AI PDF Chatbox",
    page_icon="🛰️",
    layout="centered",
    initial_sidebar_state="collapsed",
)


# =========================================================
# PROFESSIONAL UI STYLING
# =========================================================

st.markdown(
    """
    <style>

    @import url(
        'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap'
    );

    /* ------------------------------
       GLOBAL
    ------------------------------ */

    * {
        font-family: 'Inter', sans-serif;
    }

    header {
        display: none !important;
    }

    footer {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    [data-testid="stSidebarNav"] {
        display: none !important;
    }

    .stApp {
        min-height: 100vh;

        background:
        linear-gradient(
            rgba(3, 10, 28, 0.86),
            rgba(8, 10, 38, 0.90)
        ),
        url("https://images.unsplash.com/photo-1518770660439-4636190af475?auto=format&fit=crop&w=2400&q=90");

        background-size: cover;
        background-position: center;
        background-attachment: fixed;

        color: white;
    }

    .block-container {
        max-width: 500px !important;
        padding-top: 4rem !important;
        padding-bottom: 2rem !important;
    }


    /* ------------------------------
       BRAND
    ------------------------------ */

    .brand-icon {
        font-size: 42px;
        text-align: center;
        margin-bottom: 5px;
    }

    .brand-name {
        text-align: center;
        font-size: 22px;
        font-weight: 800;
        letter-spacing: 0.5px;
        color: white;
    }

    .brand-blue {
        color: #3da9ff;
    }

    .brand-description {
        text-align: center;
        font-size: 13px;
        color: rgba(255,255,255,0.60);
        margin-top: 6px;
    }


    /* ------------------------------
       WELCOME
    ------------------------------ */

    .welcome {
        text-align: center;
        margin-top: 30px;
        margin-bottom: 8px;

        font-size: 36px;
        font-weight: 800;
        letter-spacing: -1px;

        color: white;

        text-shadow:
            0 0 25px rgba(40,140,255,0.25);
    }

    .description {
        text-align: center;
        color: rgba(255,255,255,0.68);
        font-size: 14px;
        margin-bottom: 28px;
    }


    /* ------------------------------
       GOOGLE VERIFIED
    ------------------------------ */

    .verified {
        padding: 14px 16px;

        border-radius: 12px;

        background: rgba(30,120,255,0.12);

        border:
            1px solid rgba(60,160,255,0.28);

        margin-bottom: 20px;

        color: #72b8ff;

        font-size: 13px;
    }


    /* ------------------------------
       INPUTS
    ------------------------------ */

    .stTextInput label {
        color: #f1f5f9 !important;
        font-size: 13px !important;
        font-weight: 600 !important;
    }

    .stTextInput input {
        background: rgba(4,10,25,0.75) !important;

        color: white !important;

        border-radius: 11px !important;

        border:
            1px solid rgba(150,170,200,0.25) !important;

        height: 48px !important;

        padding-left: 14px !important;
    }

    .stTextInput input::placeholder {
        color: rgba(200,210,225,0.40) !important;
    }

    .stTextInput input:focus {
        border-color: #3da9ff !important;
    }


    /* ------------------------------
       BUTTON
    ------------------------------ */

    .stButton > button {
        width: 100% !important;

        height: 48px !important;

        border-radius: 11px !important;

        border: none !important;

        background:
            linear-gradient(
                90deg,
                #1597ff,
                #6255f5
            ) !important;

        color: white !important;

        font-size: 15px !important;

        font-weight: 700 !important;

        box-shadow:
            0 8px 25px rgba(40,120,255,0.25) !important;
    }

    .stButton > button:hover {
        background:
            linear-gradient(
                90deg,
                #28a5ff,
                #765cff
            ) !important;

        transform: translateY(-1px);
    }


    /* ------------------------------
       SECURITY MESSAGE
    ------------------------------ */

    .security {
        text-align: center;

        margin-top: 18px;

        color: rgba(220,230,245,0.48);

        font-size: 11px;

        line-height: 1.5;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# GOOGLE LOGIN STATUS
# =========================================================

google_logged_in = getattr(
    st.user,
    "is_logged_in",
    False
)


# =========================================================
# NOT LOGGED INTO GOOGLE
# =========================================================

if not google_logged_in:

    # -----------------------------------------------------
    # BRAND
    # -----------------------------------------------------

    st.markdown(
        '<div class="brand-icon">📄</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-name">'
        'AI PDF <span class="brand-blue">CHATBOX</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-description">'
        'Your intelligent document workspace'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # WELCOME
    # -----------------------------------------------------

    st.markdown(
        '<div class="welcome">Welcome</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="description">'
        'Sign in to continue to your personal AI workspace.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # GOOGLE LOGIN
    # -----------------------------------------------------

    if st.button(
        "🌐  Continue with Google",
        key="google_login",
        type="primary",
    ):
        st.login()


    st.markdown(
        '<div class="security">'
        '🔒 Secure authentication powered by Google'
        '</div>',
        unsafe_allow_html=True
    )

    st.stop()


# =========================================================
# GOOGLE INFORMATION
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

    st.error(
        "Unable to retrieve your Google email address."
    )

    st.stop()


# =========================================================
# CHECK USER
# =========================================================

has_account = email_exists(google_email)


# =========================================================
# NEW USER
# =========================================================

if not has_account:

    # -----------------------------------------------------
    # BRAND
    # -----------------------------------------------------

    st.markdown(
        '<div class="brand-icon">🛰️</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-name">'
        'VYPER <span class="brand-blue">AI</span>'
        '</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="brand-description">'
        'Your intelligent document workspace'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # WELCOME
    # -----------------------------------------------------

    st.markdown(
        '<div class="welcome">Welcome</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="description">'
        'Create your account to start using VYPER AI.'
        '</div>',
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # VERIFIED GOOGLE ACCOUNT
    # -----------------------------------------------------

    st.markdown(
        f"""
        <div class="verified">
            ✓ Google account verified<br>
            <strong>{google_email}</strong>
        </div>
        """,
        unsafe_allow_html=True
    )


    # -----------------------------------------------------
    # ACCOUNT CREATION
    # -----------------------------------------------------

    username = st.text_input(
        "Username",
        placeholder="Choose your username",
        key="new_username",
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Create a password",
        key="new_password",
    )

    confirm_password = st.text_input(
        "Confirm Password",
        type="password",
        placeholder="Confirm your password",
        key="confirm_password",
    )


    if st.button(
        "Create Account →",
        type="primary",
        key="create_account",
    ):

        username = username.strip()

        if not username:

            st.error(
                "Please enter a username."
            )

        elif len(username) < 3:

            st.error(
                "Username must contain at least 3 characters."
            )

        elif not password:

            st.error(
                "Please enter a password."
            )

        elif len(password) < 6:

            st.error(
                "Password must contain at least 6 characters."
            )

        elif password != confirm_password:

            st.error(
                "Passwords do not match."
            )

        elif username_exists(username):

            st.error(
                "That username is already taken."
            )

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

                st.success(
                    "Account created successfully!"
                )

                st.switch_page("app.py")

            else:

                st.error(message)


    st.markdown(
        '<div class="security">'
        '🔒 Your account is protected by Google verification '
        'and password authentication.'
        '</div>',
        unsafe_allow_html=True
    )

    st.stop()


# =========================================================
# EXISTING USER LOGIN
# =========================================================

# ---------------------------------------------------------
# BRAND
# ---------------------------------------------------------

st.markdown(
    '<div class="brand-icon">🛰️</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="brand-name">'
    'VYPER <span class="brand-blue">AI</span>'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="brand-description">'
    'Your intelligent document workspace'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# WELCOME
# ---------------------------------------------------------

st.markdown(
    '<div class="welcome">Welcome</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="description">'
    'Enter your username and password to access your workspace.'
    '</div>',
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# VERIFIED GOOGLE ACCOUNT
# ---------------------------------------------------------

st.markdown(
    f"""
    <div class="verified">
        ✓ Google account verified<br>
        <strong>{google_email}</strong>
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# LOGIN
# ---------------------------------------------------------

username = st.text_input(
    "Username",
    placeholder="Enter your username",
    key="login_username",
)

password = st.text_input(
    "Password",
    type="password",
    placeholder="Enter your password",
    key="login_password",
)


# ---------------------------------------------------------
# SIGN IN
# ---------------------------------------------------------

if st.button(
    "Sign In →",
    type="primary",
    key="signin_button",
):

    if not username:

        st.error(
            "Please enter your username."
        )

    elif not password:

        st.error(
            "Please enter your password."
        )

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

            st.success(
                "Login successful!"
            )

            st.switch_page("app.py")

        else:

            st.error(
                "Invalid username or password "
                "for this Google account."
            )


# ---------------------------------------------------------
# HOW LOGIN WORKS
# ---------------------------------------------------------

st.markdown(
    """
    <div class="security">
        🔐 <strong>How to sign in</strong><br><br>
        1. Your Google account is verified first.<br>
        2. Enter your registered username.<br>
        3. Enter your account password.<br>
        4. Click <strong>Sign In</strong> to access your workspace.
    </div>
    """,
    unsafe_allow_html=True
)