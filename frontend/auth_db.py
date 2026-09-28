"""
frontend/auth_db.py

Database functions for username/password authentication
linked to verified Google accounts.
"""

import sqlite3
import hashlib
import os


DB_PATH = os.path.join(
    os.path.dirname(__file__),
    "auth.db"
)


# =========================================================
# DATABASE CONNECTION
# =========================================================

def get_connection():
    return sqlite3.connect(DB_PATH)


# =========================================================
# CREATE TABLE
# =========================================================

def init_db():
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL
        )
        """
    )

    conn.commit()
    conn.close()


# =========================================================
# PASSWORD HASHING
# =========================================================

def hash_password(password):
    return hashlib.sha256(
        password.encode("utf-8")
    ).hexdigest()


# =========================================================
# CHECK EMAIL
# =========================================================

def email_exists(email):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM users WHERE email = ?",
        (email,)
    )

    result = cursor.fetchone()

    conn.close()

    return result is not None


# =========================================================
# CHECK USERNAME
# =========================================================

def username_exists(username):
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute(
        "SELECT 1 FROM users WHERE username = ?",
        (username,)
    )

    result = cursor.fetchone()

    conn.close()

    return result is not None


# =========================================================
# CREATE USER
# =========================================================

def create_user(username, email, password):

    if email_exists(email):
        return False, "This Google account is already registered."

    if username_exists(username):
        return False, "Username already exists."

    password_hash = hash_password(password)

    conn = get_connection()
    cursor = conn.cursor()

    try:

        cursor.execute(
            """
            INSERT INTO users
            (username, email, password_hash)
            VALUES (?, ?, ?)
            """,
            (
                username,
                email,
                password_hash
            )
        )

        conn.commit()

        return True, "Account created successfully."

    except sqlite3.IntegrityError:

        return False, "Username or email already exists."

    finally:

        conn.close()


# =========================================================
# AUTHENTICATE USER
# =========================================================

def authenticate_user(username, password, email):

    conn = get_connection()
    cursor = conn.cursor()

    password_hash = hash_password(password)

    cursor.execute(
        """
        SELECT id
        FROM users
        WHERE username = ?
        AND email = ?
        AND password_hash = ?
        """,
        (
            username,
            email,
            password_hash
        )
    )

    result = cursor.fetchone()

    conn.close()

    return result is not None


# =========================================================
# INITIALIZE DATABASE
# =========================================================

init_db()