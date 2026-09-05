"""
llm.py
==================

Member 4: LLM Integration

Supports:
    - Google Gemini
    - Groq
    - Ollama (Local)

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

    LLM_PROVIDER=ollama

    GEMINI_API_KEY=...
    GEMINI_MODEL=gemini-2.5-flash

    GROQ_API_KEY=...
    GROQ_MODEL=llama-3.3-70b-versatile
    
    OLLAMA_MODEL=llama3
    OLLAMA_BASE_URL=http://localhost:11434
"""

from __future__ import annotations

import json
import logging
import os
import requests
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


# ============================================================
# PROVIDER CONFIGURATION
# ============================================================

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

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3",
)

OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL",
    "http://localhost:11434",
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

print("OLLAMA_MODEL:", OLLAMA_MODEL)
print("OLLAMA_BASE_URL:", OLLAMA_BASE_URL)

print("ENV FILE:", ENV_FILE)
print("ENV EXISTS:", ENV_FILE.exists())

print("========================================")


# ============================================================
# CONFIGURATION
# ============================================================

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
    "ollama": os.getenv(
        "OLLAMA_MODEL",
        "llama3",
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
    format=(
        "%(asctime)s | "
        "%(levelname)s | "
        "%(message)s"
    ),
)

logger = logging.getLogger(
    "llm_integration"
)


# ============================================================
# PROVIDER VALIDATION
# ============================================================

SUPPORTED_PROVIDERS = {
    "gemini",
    "groq",
    "ollama",
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

    # Ollama is local and does not require an API key
    if PROVIDER == "ollama":
        return

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
# OLLAMA (LOCAL)
# ============================================================

def _stream_ollama(
    prompt: str,
) -> Generator[str, None, None]:
    """
    Stream response from a local Ollama instance.
    Requires the Ollama app to be running.
    """
    url = f"{OLLAMA_BASE_URL.rstrip('/')}/api/generate"
    payload = {
        "model": MODELS["ollama"],
        "prompt": prompt,
        "stream": True,
        "options": {
            "num_predict": MAX_OUTPUT_TOKENS
        }
    }

    try:
        with requests.post(url, json=payload, stream=True) as response:
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                    if data.get("done"):
                        break
                        
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            f"Could not connect to Ollama at {OLLAMA_BASE_URL}. "
            "Please ensure the Ollama application is running locally."
        )
    except Exception as error:
        raise RuntimeError(f"Ollama generation failed: {error}")


# ============================================================
# GOOGLE GEMINI (OPTIMIZED TRUE STREAMING)
# ============================================================

from google import genai

def _stream_gemini(
    prompt: str,
) -> Generator[str, None, None]:
    """
    Stream response chunks directly from Google Gemini as they are generated.
    Enables low-latency token-by-token streaming to the client.
    """

    client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    try:
        response = client.models.generate_content_stream(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        has_yielded = False
        for chunk in response:
            if chunk.text:
                has_yielded = True
                yield chunk.text

        if not has_yielded:
            raise RuntimeError(
                f"Gemini returned no text for model '{GEMINI_MODEL}'. "
                "The response may have been empty or blocked by safety filters."
            )

    except Exception as error:
        raise RuntimeError(
            f"Gemini API streaming failed for model "
            f"'{GEMINI_MODEL}': {error}"
        ) from error


# ============================================================
# GROQ
# ============================================================

def _stream_groq(
    prompt: str,
) -> Generator[str, None, None]:
    """
    Stream response from Groq.
    """

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
        if (
            chunk.choices
            and chunk.choices[0].delta
            and chunk.choices[0].delta.content
        ):
            yield chunk.choices[0].delta.content


# ============================================================
# COMMON STREAMING INTERFACE
# ============================================================

def stream_llm_response(
    prompt: str,
) -> Generator[str, None, None]:
    """
    Main LLM interface.

    The LLM is NOT restricted to retrieved context.
    It can answer using:
        1. PDF context
        2. Knowledge-base context
        3. General pretrained knowledge

    The actual source-selection instructions are supplied
    by api_framework.py through the final prompt.
    """

    # --------------------------------------------------------
    # Validate prompt
    # --------------------------------------------------------

    if not isinstance(
        prompt,
        str,
    ):
        raise ValueError(
            "Prompt must be a string."
        )

    prompt = prompt.strip()

    if not prompt:
        raise ValueError(
            "Prompt cannot be empty."
        )

    # --------------------------------------------------------
    # Validate provider
    # --------------------------------------------------------

    validate_configuration()

    model = MODELS[PROVIDER]

    logger.info(
        "LLM request | provider=%s | model=%s",
        PROVIDER,
        model,
    )

    try:
        if PROVIDER == "gemini":
            yield from _stream_gemini(
                prompt
            )

        elif PROVIDER == "groq":
            yield from _stream_groq(
                prompt
            )

        elif PROVIDER == "ollama":
            yield from _stream_ollama(
                prompt
            )

    except Exception as exc:
        logger.exception(
            "LLM request failed: %s",
            exc,
        )

        yield (
            "\n\nSorry, the AI service is "
            "temporarily unavailable. "
            f"({exc})"
        )


# ============================================================
# PROVIDER INFORMATION
# ============================================================

def get_llm_info() -> dict[str, str]:
    """
    Return current provider configuration.
    """

    return {
        "provider": PROVIDER,
        "model": MODELS[PROVIDER],
    }


# ============================================================
# LOCAL TEST
# ============================================================

def _run_test() -> None:
    """
    Test the currently selected LLM provider.
    """

    print("=" * 65)
    print(
        "LLM INTEGRATION TEST"
    )
    print("=" * 65)

    validate_configuration()

    info = get_llm_info()

    print(
        f"Provider : {info['provider']}"
    )
    print(
        f"Model    : {info['model']}"
    )

    test_prompt = """
You are VYPER, a professional AI assistant.

You have access to retrieved context, but retrieved
context is not a restriction on your knowledge.

If the context contains the answer, use it.

If the context does not contain the answer and the
question is a general knowledge question, use your
general pretrained knowledge.

Do not invent facts.

Context:
Artificial Intelligence is the field of computer
science concerned with creating systems capable of
performing tasks that normally require human intelligence.

Question:
What is an LLM?

Answer:
"""

    print(
        "\nAI Response:"
    )

    for chunk in stream_llm_response(
        test_prompt
    ):
        print(
            chunk,
            end="",
            flush=True,
        )

    print(
        "\n"
    )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":
    _run_test()