"""
memory.py
=========

Week 3 - Member 3: Conversation Memory

Uses LangChain's ConversationBufferWindowMemory.

The application maintains a separate memory object for every
session_id.

The project requirement is to retain the latest 5 messages.
Therefore, this wrapper explicitly enforces a 5-message window
after every message is added.

NOTE:
    ConversationBufferWindowMemory is a legacy/deprecated
    LangChain API, but it is intentionally used here because
    it is part of the project requirement.

    Memory is stored in RAM and will be cleared when the API
    server is restarted.
"""

from __future__ import annotations

from threading import Lock
from typing import Dict, List

from langchain_classic.memory import (
    ConversationBufferWindowMemory,
)


# ============================================================
# CONFIGURATION
# ============================================================

MEMORY_WINDOW = 5


# ============================================================
# CONVERSATION MEMORY MANAGER
# ============================================================

class ConversationMemory:
    """
    Manages conversation memory for multiple sessions.

    Each session gets its own ConversationBufferWindowMemory.

    Example:

        session_1
            -> User
            -> Assistant
            -> User
            -> Assistant

        session_2
            -> User
            -> Assistant
    """

    def __init__(
        self,
        window_size: int = MEMORY_WINDOW,
    ) -> None:

        if window_size <= 0:
            raise ValueError(
                "window_size must be greater than 0."
            )

        self.window_size = window_size

        # ----------------------------------------------------
        # session_id -> LangChain memory object
        # ----------------------------------------------------

        self._sessions: Dict[
            str,
            ConversationBufferWindowMemory,
        ] = {}

        # ----------------------------------------------------
        # Protect memory when multiple API requests happen
        # simultaneously.
        # ----------------------------------------------------

        self._lock = Lock()


    # ========================================================
    # VALIDATE SESSION ID
    # ========================================================

    @staticmethod
    def _validate_session_id(
        session_id: str,
    ) -> str:
        """
        Validate and normalize a session ID.
        """

        if not isinstance(
            session_id,
            str,
        ):
            raise TypeError(
                "session_id must be a string."
            )

        session_id = session_id.strip()

        if not session_id:
            raise ValueError(
                "session_id cannot be empty."
            )

        return session_id


    # ========================================================
    # GET OR CREATE MEMORY
    # ========================================================

    def _get_memory(
        self,
        session_id: str,
    ) -> ConversationBufferWindowMemory:
        """
        Return the LangChain memory associated with a session.

        If the session does not exist, create it.
        """

        session_id = self._validate_session_id(
            session_id
        )

        with self._lock:

            if session_id not in self._sessions:

                self._sessions[
                    session_id
                ] = ConversationBufferWindowMemory(
                    k=self.window_size,
                    memory_key="history",
                    return_messages=True,
                )

            return self._sessions[
                session_id
            ]


    # ========================================================
    # TRIM HISTORY
    # ========================================================

    def _trim_history(
        self,
        memory: ConversationBufferWindowMemory,
    ) -> None:
        """
        Explicitly keep only the latest window_size messages.

        This is done because the project requirement is
        specifically 'last 5 messages'.
        """

        messages = memory.chat_memory.messages

        if len(messages) <= self.window_size:
            return

        memory.chat_memory.messages = (
            messages[-self.window_size:]
        )


    # ========================================================
    # ADD USER MESSAGE
    # ========================================================

    def add_user_message(
        self,
        session_id: str,
        content: str,
    ) -> None:
        """
        Add a user message and enforce the memory window.
        """

        if not isinstance(
            content,
            str,
        ):
            raise TypeError(
                "content must be a string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "User message cannot be empty."
            )

        memory = self._get_memory(
            session_id
        )

        with self._lock:

            memory.chat_memory.add_user_message(
                content
            )

            self._trim_history(
                memory
            )


    # ========================================================
    # ADD ASSISTANT MESSAGE
    # ========================================================

    def add_assistant_message(
        self,
        session_id: str,
        content: str,
    ) -> None:
        """
        Add an assistant message and enforce the memory window.
        """

        if not isinstance(
            content,
            str,
        ):
            raise TypeError(
                "content must be a string."
            )

        content = content.strip()

        if not content:
            raise ValueError(
                "Assistant message cannot be empty."
            )

        memory = self._get_memory(
            session_id
        )

        with self._lock:

            memory.chat_memory.add_ai_message(
                content
            )

            self._trim_history(
                memory
            )


    # ========================================================
    # GET RECENT HISTORY
    # ========================================================

    def get_recent_history(
        self,
        session_id: str,
    ) -> List:
        """
        Return the latest 5 LangChain messages.
        """

        memory = self._get_memory(
            session_id
        )

        with self._lock:

            self._trim_history(
                memory
            )

            return list(
                memory.chat_memory.messages
            )


    # ========================================================
    # GET HISTORY AS TEXT
    # ========================================================

    def get_history_text(
        self,
        session_id: str,
    ) -> str:
        """
        Return conversation history as readable text.

        Example:

            User: What is RAG?
            Assistant: RAG is Retrieval-Augmented Generation.
            User: What are its advantages?
        """

        history = self.get_recent_history(
            session_id
        )

        if not history:
            return ""

        lines = []

        for message in history:

            if message.type == "human":

                role = "User"

            elif message.type == "ai":

                role = "Assistant"

            else:

                role = message.type.capitalize()

            lines.append(
                f"{role}: {message.content}"
            )

        return "\n".join(
            lines
        )


    # ========================================================
    # MESSAGE COUNT
    # ========================================================

    def message_count(
        self,
        session_id: str,
    ) -> int:
        """
        Return the number of messages currently stored.
        """

        return len(
            self.get_recent_history(
                session_id
            )
        )


    # ========================================================
    # CHECK SESSION
    # ========================================================

    def has_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Check whether a session exists.
        """

        if not isinstance(
            session_id,
            str,
        ):
            return False

        session_id = session_id.strip()

        with self._lock:

            return (
                session_id
                in self._sessions
            )


    # ========================================================
    # CLEAR SESSION
    # ========================================================

    def clear_session(
        self,
        session_id: str,
    ) -> bool:
        """
        Delete one session's conversation.
        """

        if not isinstance(
            session_id,
            str,
        ):
            return False

        session_id = session_id.strip()

        with self._lock:

            if session_id in self._sessions:

                del self._sessions[
                    session_id
                ]

                return True

        return False


    # ========================================================
    # CLEAR ALL SESSIONS
    # ========================================================

    def clear_all(self) -> None:
        """
        Delete every conversation.
        """

        with self._lock:

            self._sessions.clear()


# ============================================================
# GLOBAL MEMORY INSTANCE
# ============================================================

conversation_memory = ConversationMemory(
    window_size=MEMORY_WINDOW
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_user_message(
    session_id: str,
    content: str,
) -> None:
    """
    Add a user message.
    """

    conversation_memory.add_user_message(
        session_id=session_id,
        content=content,
    )


def add_assistant_message(
    session_id: str,
    content: str,
) -> None:
    """
    Add an assistant message.
    """

    conversation_memory.add_assistant_message(
        session_id=session_id,
        content=content,
    )


def get_recent_history(
    session_id: str,
):
    """
    Get recent LangChain messages.
    """

    return conversation_memory.get_recent_history(
        session_id=session_id
    )


def get_history_text(
    session_id: str,
) -> str:
    """
    Get recent history as plain text.
    """

    return conversation_memory.get_history_text(
        session_id=session_id
    )


def clear_session(
    session_id: str,
) -> bool:
    """
    Clear a conversation session.
    """

    return conversation_memory.clear_session(
        session_id
    )


# ============================================================
# TESTS
# ============================================================

def run_tests() -> None:
    """
    Test Member 3 conversation memory.
    """

    print()
    print("=" * 70)
    print("LANGCHAIN CONVERSATION MEMORY TEST")
    print("=" * 70)

    memory = ConversationMemory(
        window_size=5
    )

    session_id = "test-session"

    # --------------------------------------------------------
    # Add 6 messages
    # --------------------------------------------------------

    memory.add_user_message(
        session_id,
        "What is artificial intelligence?",
    )

    memory.add_assistant_message(
        session_id,
        "Artificial intelligence is a field of computer science.",
    )

    memory.add_user_message(
        session_id,
        "What are its applications?",
    )

    memory.add_assistant_message(
        session_id,
        "AI is used in healthcare, finance and robotics.",
    )

    memory.add_user_message(
        session_id,
        "Which one is used in healthcare?",
    )

    memory.add_assistant_message(
        session_id,
        "AI is used for medical imaging and diagnosis.",
    )

    # --------------------------------------------------------
    # Get history
    # --------------------------------------------------------

    history = memory.get_recent_history(
        session_id
    )

    print()
    print(
        f"Messages stored: {len(history)}"
    )

    print()
    print("Recent history:")

    for index, message in enumerate(
        history,
        start=1,
    ):

        print(
            f"{index}. "
            f"{message.type}: "
            f"{message.content}"
        )

    # --------------------------------------------------------
    # Test exactly 5 messages
    # --------------------------------------------------------

    assert len(history) == 5, (
        f"Expected 5 messages, "
        f"got {len(history)}"
    )

    # --------------------------------------------------------
    # First message must have disappeared
    # --------------------------------------------------------

    assert all(
        message.content
        != "What is artificial intelligence?"
        for message in history
    )

    # --------------------------------------------------------
    # Latest message must exist
    # --------------------------------------------------------

    assert (
        history[-1].content
        == "AI is used for medical imaging and diagnosis."
    )

    # --------------------------------------------------------
    # Test text formatting
    # --------------------------------------------------------

    history_text = memory.get_history_text(
        session_id
    )

    print()
    print("Formatted history:")
    print("-" * 70)
    print(history_text)
    print("-" * 70)

    assert "User:" in history_text
    assert "Assistant:" in history_text

    # --------------------------------------------------------
    # Test session isolation
    # --------------------------------------------------------

    second_session = "session-two"

    memory.add_user_message(
        second_session,
        "This is another conversation.",
    )

    assert (
        memory.message_count(
            second_session
        )
        == 1
    )

    assert (
        memory.message_count(
            session_id
        )
        == 5
    )

    # --------------------------------------------------------
    # Test clearing
    # --------------------------------------------------------

    assert memory.clear_session(
        session_id
    ) is True

    assert (
        memory.has_session(
            session_id
        )
        is False
    )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("ALL LANGCHAIN MEMORY TESTS PASSED")
    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    run_tests()