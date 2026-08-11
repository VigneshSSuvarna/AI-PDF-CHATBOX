"""
llm_integration.py
==================

Member 4: LLM Integration

Supports:
    - Google Gemini
    - Anthropic Claude
    - Groq

Responsibilities:
    - Receive the final RAG prompt
    - Call the selected LLM
    - Stream generated text
    - Provide a common interface for Member 1
    - Handle provider errors cleanly

Member 4 does NOT handle:
    - PDF processing
    - Chunking
    - Embeddings
    - ChromaDB
    - Retrieval
    - Metadata filtering
    - Conversation memory
    - Prompt construction

Environment variables:

    LLM_PROVIDER=groq

    GEMINI_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash

    ANTHROPIC_API_KEY=...
    CLAUDE_MODEL=claude-sonnet-4-20250514

    GROQ_API_KEY=...
    GROQ_MODEL=llama-3.3-70b-versatile
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator


# ============================================================
# CONFIGURATION
# ============================================================

PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "groq",
).strip().lower()

MODELS = {
    "gemini": os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
    ),
    "claude": os.getenv(
        "CLAUDE_MODEL",
        "claude-sonnet-4-20250514",
    ),
    "groq": os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile",
    ),
}

MAX_OUTPUT_TOKENS = int(
    os.getenv(
        "LLM_MAX_OUTPUT_TOKENS",
        "1000",
    )
)


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger("llm_integration")


# ============================================================
# PROVIDER VALIDATION
# ============================================================

SUPPORTED_PROVIDERS = {
    "gemini",
    "claude",
    "groq",
}


def validate_configuration() -> None:
    """Validate the selected provider and API key."""

    if PROVIDER not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported LLM_PROVIDER='{PROVIDER}'. "
            f"Choose from: {sorted(SUPPORTED_PROVIDERS)}"
        )

    if MAX_OUTPUT_TOKENS <= 0:
        raise ValueError(
            "LLM_MAX_OUTPUT_TOKENS must be greater than 0."
        )

    key_names = {
        "gemini": "GEMINI_API_KEY",
        "claude": "ANTHROPIC_API_KEY",
        "groq": "GROQ_API_KEY",
    }

    key_name = key_names[PROVIDER]

    if not os.getenv(key_name):
        raise RuntimeError(
            f"{key_name} is not configured."
        )


# ============================================================
# GOOGLE GEMINI
# ============================================================

def _stream_gemini(
    prompt: str,
) -> Generator[str, None, None]:
    """Stream response from Google Gemini."""

    from google import genai

    client = genai.Client(
        api_key=os.environ["GEMINI_API_KEY"]
    )

    response = client.models.generate_content_stream(
        model=MODELS["gemini"],
        contents=prompt,
        config={
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },
    )

    for chunk in response:

        if chunk.text:
            yield chunk.text


# ============================================================
# ANTHROPIC CLAUDE
# ============================================================

def _stream_claude(
    prompt: str,
) -> Generator[str, None, None]:
    """Stream response from Anthropic Claude."""

    from anthropic import Anthropic

    client = Anthropic(
        api_key=os.environ["ANTHROPIC_API_KEY"]
    )

    with client.messages.stream(
        model=MODELS["claude"],
        max_tokens=MAX_OUTPUT_TOKENS,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
    ) as stream:

        for text in stream.text_stream:
            if text:
                yield text


# ============================================================
# GROQ
# ============================================================

def _stream_groq(
    prompt: str,
) -> Generator[str, None, None]:
    """Stream response from Groq."""

    from groq import Groq

    client = Groq(
        api_key=os.environ["GROQ_API_KEY"]
    )

    stream = client.chat.completions.create(
        model=MODELS["groq"],
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    )

    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


# ============================================================
# COMMON STREAMING INTERFACE
# ============================================================

def stream_llm_response(
    prompt: str,
) -> Generator[str, None, None]:
    """
    Main interface used by Member 1.

    The rest of the application only needs:

        stream_llm_response(prompt)

    The selected provider is handled internally.
    """

    if not isinstance(prompt, str):
        raise ValueError(
            "Prompt must be a string."
        )

    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            "Prompt cannot be empty."
        )

    validate_configuration()

    model = MODELS[PROVIDER]

    logger.info(
        "LLM request | provider=%s | model=%s",
        PROVIDER,
        model,
    )

    try:

        if PROVIDER == "gemini":

            yield from _stream_gemini(prompt)

        elif PROVIDER == "claude":

            yield from _stream_claude(prompt)

        elif PROVIDER == "groq":

            yield from _stream_groq(prompt)

    except Exception as exc:

        logger.exception(
            "LLM request failed: %s",
            exc,
        )

        yield (
            "\n\nSorry, the AI service is "
            "temporarily unavailable."
        )


# ============================================================
# PROVIDER INFORMATION
# ============================================================

def get_llm_info() -> dict[str, str]:
    """Return current provider configuration."""

    return {
        "provider": PROVIDER,
        "model": MODELS[PROVIDER],
    }


# ============================================================
# LOCAL TEST
# ============================================================

def _run_test() -> None:
    """Test the currently selected LLM provider."""

    print("=" * 65)
    print("LLM INTEGRATION TEST")
    print("=" * 65)

    validate_configuration()

    info = get_llm_info()

    print(f"Provider : {info['provider']}")
    print(f"Model    : {info['model']}")

    test_prompt = """
You are answering a question in an AI PDF Chatbox.

Context:
Artificial Intelligence is the field of computer
science concerned with creating systems capable of
performing tasks that normally require human intelligence.

Question:
What is Artificial Intelligence?

Answer using the provided context.
"""

    print("\nAI Response:")
    print("-" * 65)

    for text in stream_llm_response(test_prompt):

        print(
            text,
            end="",
            flush=True,
        )

    print("\n")
    print("=" * 65)
    print("TEST COMPLETED")
    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    _run_test()