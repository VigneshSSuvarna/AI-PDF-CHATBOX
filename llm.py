"""
llm.py
==================

Member 4: LLM Integration

Supports:
    - Google Gemini
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

    LLM_PROVIDER=gemini  # Switched default to gemini

    GEMINI_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash

    GROQ_API_KEY=...
    GROQ_MODEL=llama-3.3-70b-versatile
"""

from __future__ import annotations

import logging
import os
from collections.abc import Generator

from pathlib import Path

from dotenv import load_dotenv


# ============================================================
# LOAD .ENV
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

load_dotenv(
    dotenv_path=ENV_FILE,
    override=False,
)

# 🔥 FIX: Changed default to "gemini"
LLM_PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini",
).lower()

GROQ_API_KEY = os.getenv(
    "GROQ_API_KEY"
)

GROQ_MODEL = os.getenv(
    "GROQ_MODEL",
    "llama-3.3-70b-versatile",
)

GEMINI_API_KEY = os.getenv(
    "GEMINI_API_KEY"
)

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash",
)

print("========================================")
print("LLM CONFIGURATION")
print("========================================")
print("LLM_PROVIDER:", LLM_PROVIDER)
print(
    "GROQ_API_KEY:",
    "CONFIGURED" if GROQ_API_KEY else "NOT CONFIGURED"
)
print("GROQ_MODEL:", GROQ_MODEL)
print(
    "GEMINI_API_KEY:",
    "CONFIGURED" if GEMINI_API_KEY else "NOT CONFIGURED"
)
print("GEMINI_MODEL:", GEMINI_MODEL)
print("ENV FILE:", ENV_FILE)
print("ENV EXISTS:", ENV_FILE.exists())
print("========================================")

#============================================================
# CONFIGURATION
# ============================================================

# 🔥 FIX: Changed default to "gemini" here as well
PROVIDER = os.getenv(
    "LLM_PROVIDER",
    "gemini",
).strip().lower()

MODELS = {
    "gemini": os.getenv(
        "GEMINI_MODEL",
        "gemini-2.5-flash",
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

from google import genai


def _stream_gemini(
    prompt: str,
) -> Generator[str, None, None]:
    """
    Stream response from Google Gemini.

    NOTE: The google-genai SDK's generate_content() is NOT a
    streaming call here — it returns the full response in one
    shot. We yield it as a single chunk once it's ready.
    """

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )
    except Exception as error:
        raise RuntimeError(
            f"Gemini API call failed for model "
            f"'{GEMINI_MODEL}': {error}"
        ) from error

    if response.text:
        yield response.text
        return

    finish_reason = None
    safety_ratings = None

    if getattr(response, "candidates", None):
        candidate = response.candidates[0]
        finish_reason = getattr(candidate, "finish_reason", None)
        safety_ratings = getattr(candidate, "safety_ratings", None)

    prompt_feedback = getattr(response, "prompt_feedback", None)

    raise RuntimeError(
        "Gemini returned no text. "
        f"model='{GEMINI_MODEL}', "
        f"finish_reason={finish_reason}, "
        f"prompt_feedback={prompt_feedback}, "
        f"safety_ratings={safety_ratings}. "
        "Check that GEMINI_MODEL is a valid, currently "
        "supported model name and that the prompt was not "
        "blocked by safety filters."
    )


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

        elif PROVIDER == "groq":

            yield from _stream_groq(prompt)

    except Exception as exc:

        logger.exception(
            "LLM request failed: %s",
            exc,
        )

        yield (
            f"\n\nSorry, the AI service is temporarily "
            f"unavailable. ({exc})"
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