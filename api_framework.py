"""
member1_api_framework.py
========================

Week 3 - Member 1: API Framework

FastAPI backend for the AI PDF Chatbox.

Responsibilities:
    - Expose /chat endpoint
    - Expose /upload endpoint
    - Connect the API to the existing retriever
    - Provide health check
    - Validate uploaded PDFs
    - Provide error handling
    - Provide Swagger API documentation

Current integrations:
    - Retriever -> CONNECTED
    - Member 2 Prompt Engineering -> integration point
    - Member 3 Conversation Memory -> integration point
    - Member 4 LLM Integration -> integration point

Run:
    pip install fastapi uvicorn[standard] pydantic python-multipart
    uvicorn member1_api_framework:app --reload

Open:
    http://127.0.0.1:8000/docs

Self-test:
    python api_framework.py
"""

from __future__ import annotations

import logging
import uuid

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Request,
    UploadFile,
    status,
)

from fastapi.middleware.cors import CORSMiddleware

from fastapi.responses import (
    JSONResponse,
    StreamingResponse,
)

from pydantic import BaseModel, Field


# ============================================================
# IMPORT YOUR WEEK 2 RETRIEVER
# ============================================================

from retriever import Retriever

from memory import (
    get_history_text,
    add_user_message,
    add_assistant_message,
)


# ============================================================
# CONFIGURATION
# ============================================================

UPLOAD_DIR = Path("uploads")

ALLOWED_EXTENSIONS = {".pdf"}

MAX_FILE_SIZE_MB = 25

HISTORY_LIMIT = 5

TOP_K = 5

CORS_ALLOW_ORIGINS = ["*"]


# ============================================================
# LOGGING
# ============================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("member1_api")


# ============================================================
# RETRIEVER INITIALIZATION
# ============================================================

retriever: Retriever | None = None


def get_retriever() -> Retriever:
    """
    Create the retriever once and reuse it.

    Loading the embedding model every time /chat is called
    would be inefficient, so we initialize it once.
    """

    global retriever

    if retriever is None:

        logger.info(
            "Initializing RAG retriever..."
        )

        retriever = Retriever()

        logger.info(
            "RAG retriever initialized successfully."
        )

    return retriever


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):

    # --------------------------------------------------------
    # Create upload directory
    # --------------------------------------------------------

    UPLOAD_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    logger.info(
        "Starting RAG PDF Chatbot API"
    )

    # --------------------------------------------------------
    # Initialize retriever
    # --------------------------------------------------------

    try:

        get_retriever()

        logger.info(
            "Retriever ready."
        )

    except Exception as error:

        logger.warning(
            "Retriever could not be initialized: %s",
            error,
        )

        logger.warning(
            "The API will still start, but /chat retrieval "
            "will not work until ChromaDB is available."
        )

    yield

    logger.info(
        "Stopping RAG PDF Chatbot API"
    )


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="RAG PDF Chatbot API",
    description=(
        "Backend API for the AI PDF Chatbox "
        "RAG system."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ALLOW_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# REQUEST / RESPONSE SCHEMAS
# ============================================================

class ChatRequest(BaseModel):
    """
    Request body for /chat.
    """

    session_id: str = Field(
        ...,
        min_length=1,
        description="Conversation/session identifier",
    )

    message: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="User's question",
    )

    doc_id: str | None = Field(
        None,
        description=(
            "Optional document ID. "
            "Used when document-specific retrieval "
            "is available."
        ),
    )


class UploadResponse(BaseModel):
    """
    Response returned after uploading a PDF.
    """

    doc_id: str

    filename: str

    status: str

    uploaded_at: str


class ErrorResponse(BaseModel):
    """
    Standard API error response.
    """

    detail: str


class HealthResponse(BaseModel):
    """
    Health check response.
    """

    status: str

    version: str

    retriever: str


# ============================================================
# MEMBER 3 - CONVERSATION MEMORY
# ============================================================

def get_recent_history(
    session_id: str,
    limit: int = HISTORY_LIMIT,
) -> list[dict[str, str]]:
    """
    Member 3 conversation memory integration.

    Returns the recent conversation history in the format
    expected by the prompt builder.
    """

    history_text = get_history_text(
        session_id
    )

    if not history_text:
        return []

    history = []

    for line in history_text.splitlines():

        if line.startswith("User: "):

            history.append(
                {
                    "role": "user",
                    "content": line[
                        len("User: "):
                    ],
                }
            )

        elif line.startswith("Assistant: "):

            history.append(
                {
                    "role": "assistant",
                    "content": line[
                        len("Assistant: "):
                    ],
                }
            )

    return history[-limit:]


# ============================================================
# WEEK 2 RETRIEVER - CONNECTED
# ============================================================

def retrieve_context(
    question: str,
    doc_id: str | None = None,
) -> list[str]:
    """
    Retrieve the most relevant chunks from ChromaDB.

    This function connects the Week 3 FastAPI backend
    directly to the Week 2 Retriever.

    Parameters
    ----------
    question:
        User's question.

    doc_id:
        Optional uploaded document ID.

        NOTE:
        Document-specific filtering requires the same
        doc_id to exist in the metadata of the chunks
        stored in ChromaDB.

    Returns
    -------
    list[str]
        Retrieved document chunks.
    """

    # --------------------------------------------------------
    # Get existing Retriever instance
    # --------------------------------------------------------

    rag_retriever = get_retriever()

    # --------------------------------------------------------
    # If no document ID is supplied:
    #
    # Search the complete knowledge base.
    # --------------------------------------------------------

    if not doc_id:

        documents = rag_retriever.search(
            query=question,
            top_k=TOP_K,
        )

    # --------------------------------------------------------
    # If document ID is supplied:
    #
    # Search only chunks whose metadata contains:
    #
    #     "doc_id": <uploaded document ID>
    #
    # This requires the ingestion pipeline to store doc_id
    # inside chunk metadata.
    # --------------------------------------------------------

    else:

        metadata_filter = {
            "doc_id": doc_id
        }

        documents = rag_retriever.search(
            query=question,
            top_k=TOP_K,
            metadata_filter=metadata_filter,
        )

    # --------------------------------------------------------
    # Convert LangChain Documents into plain strings
    # --------------------------------------------------------

    context_chunks = []

    for document in documents:

        if document.page_content:

            context_chunks.append(
                document.page_content
            )

    logger.info(
        "Retrieved %d chunks for question: %s",
        len(context_chunks),
        question,
    )

    return context_chunks


# ============================================================
# MEMBER 2 - PROMPT ENGINEERING
# ============================================================

def build_prompt(
    context_chunks: list[str],
    history: list[dict[str, str]],
    question: str,
) -> str:
    """
    Member 2 integration point.

    The final version should use a strict RAG prompt that:

        1. Uses only retrieved context.
        2. Uses conversation history for follow-up questions.
        3. Does not invent information.
        4. Says "I don't know" when the answer is not
           present in the retrieved context.
        5. Includes source information where appropriate.
    """

    # --------------------------------------------------------
    # TEMPORARY DEVELOPMENT PROMPT
    #
    # This allows us to test the API/retriever connection
    # before Member 2 finishes the final prompt.
    # --------------------------------------------------------

    context = "\n\n".join(
        context_chunks
    )

    history_text = "\n".join(
        f"{message.get('role', 'user')}: "
        f"{message.get('content', '')}"
        for message in history
    )

    prompt = f"""
You are an AI assistant that answers questions using
retrieved document context.

Use the retrieved context to answer the question.

If the answer cannot be found in the context,
say that you do not know.

Do not invent facts.

Conversation history:
{history_text}

Retrieved context:
{context}

Current question:
{question}

Answer:
"""

    return prompt.strip()


# ============================================================
# MEMBER 4 - LLM INTEGRATION
# ============================================================

async def stream_llm_response(
    prompt: str,
) -> AsyncIterator[str]:
    """
    Member 4 integration point.

    This function will eventually call OpenAI, Gemini,
    Claude, Groq, etc. and stream the response.

    Expected output format:

        data: token

        data: token

        data: [DONE]
    """

    # --------------------------------------------------------
    # TEMPORARY DEVELOPMENT RESPONSE
    #
    # Remove this once Member 4 connects the actual LLM.
    # --------------------------------------------------------

    yield (
        "data: LLM integration is not connected yet. "
        "Retrieved context was successfully prepared.\n\n"
    )

    yield "data: [DONE]\n\n"


# ============================================================
# DOCUMENT INGESTION
# ============================================================

def ingest_document(
    doc_id: str,
    path: Path,
) -> None:
    """
    Document ingestion integration point.

    Eventually this should connect the uploaded PDF to
    the existing Week 1 + Week 2 pipeline:

        PDF
         ↓
        Cleaning
         ↓
        Chunking
         ↓
        Metadata
         ↓
        Embeddings
         ↓
        ChromaDB
    """

    # --------------------------------------------------------
    # TODO:
    # Connect this to the team's ingestion pipeline.
    # --------------------------------------------------------

    raise NotImplementedError(
        "Document ingestion pipeline is not connected yet."
    )


# ============================================================
# HEALTH ENDPOINT
# ============================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
)
async def health() -> HealthResponse:
    """
    Check whether the API is running.
    """

    try:

        get_retriever()

        retriever_status = "ready"

    except Exception:

        retriever_status = "unavailable"

    return HealthResponse(
        status="ok",
        version=app.version,
        retriever=retriever_status,
    )


# ============================================================
# UPLOAD ENDPOINT
# ============================================================

@app.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {
            "model": ErrorResponse
        }
    },
    tags=["upload"],
)
async def upload_document(
    file: UploadFile = File(...)
) -> UploadResponse:
    """
    Upload a PDF document.

    The PDF is validated and saved locally.

    The actual ingestion/indexing pipeline will be connected
    through ingest_document().
    """

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing filename.",
        )

    # --------------------------------------------------------
    # Validate extension
    # --------------------------------------------------------

    ext = Path(
        file.filename
    ).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Unsupported file type '{ext}'. "
                f"Allowed: {sorted(ALLOWED_EXTENSIONS)}."
            ),
        )

    # --------------------------------------------------------
    # Read file
    # --------------------------------------------------------

    contents = await file.read()

    size_mb = (
        len(contents)
        / (1024 * 1024)
    )

    # --------------------------------------------------------
    # Empty file
    # --------------------------------------------------------

    if size_mb == 0:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file is empty.",
        )

    # --------------------------------------------------------
    # Maximum size
    # --------------------------------------------------------

    if size_mb > MAX_FILE_SIZE_MB:

        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"File exceeds "
                f"{MAX_FILE_SIZE_MB}MB limit "
                f"({size_mb:.1f}MB)."
            ),
        )

    # --------------------------------------------------------
    # Generate document ID
    # --------------------------------------------------------

    doc_id = str(
        uuid.uuid4()
    )

    # --------------------------------------------------------
    # Save uploaded file
    # --------------------------------------------------------

    dest = (
        UPLOAD_DIR
        / f"{doc_id}{ext}"
    )

    dest.write_bytes(
        contents
    )

    logger.info(
        "Uploaded %s -> %s (%.2fMB)",
        file.filename,
        dest,
        size_mb,
    )

    # --------------------------------------------------------
    # Run ingestion
    # --------------------------------------------------------

    ingestion_status = "received"

    try:

        ingest_document(
            doc_id,
            dest,
        )

        ingestion_status = "indexed"

    except NotImplementedError as error:

        logger.warning(
            "Ingestion not connected yet: %s",
            error,
        )

        ingestion_status = "received"

    except Exception as error:

        logger.exception(
            "Document ingestion failed."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document ingestion failed.",
        ) from error

    # --------------------------------------------------------
    # Response
    # --------------------------------------------------------

    return UploadResponse(
        doc_id=doc_id,
        filename=file.filename,
        status=ingestion_status,
        uploaded_at=(
            datetime.now(
                timezone.utc
            ).isoformat()
        ),
    )


# ============================================================
# CHAT ENDPOINT
# ============================================================

@app.post(
    "/chat",
    responses={
        501: {
            "model": ErrorResponse
        }
    },
    tags=["chat"],
)
async def chat(
    req: ChatRequest,
) -> StreamingResponse:
    """
    Main RAG chat endpoint.

    Pipeline:

        Question
            ↓
        Conversation Memory
            ↓
        Retriever
            ↓
        Retrieved Context
            ↓
        Prompt
            ↓
        LLM
            ↓
        Streaming Response
    """

    try:

        # ----------------------------------------------------
        # 1. Conversation memory
        # ----------------------------------------------------

        history = get_recent_history(
            req.session_id,
            limit=HISTORY_LIMIT,
        )

        # ----------------------------------------------------
        # 2. Retrieve relevant chunks
        # ----------------------------------------------------

        context_chunks = retrieve_context(
            req.message,
            doc_id=req.doc_id,
        )

        # ----------------------------------------------------
        # 3. Build prompt
        # ----------------------------------------------------

        prompt = build_prompt(
            context_chunks,
            history,
            req.message,
        )

        add_user_message(
            session_id=req.session_id,
            content=req.message,
        )
        
        logger.info(
            "Prompt prepared for session %s",
            req.session_id,
        )

    except NotImplementedError as error:

        logger.warning(
            "Chat blocked by unimplemented dependency: %s",
            error,
        )

        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=str(error),
        ) from error

    except Exception as error:

        logger.exception(
            "Error preparing chat request."
        )

        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to prepare chat request.",
        ) from error

    # --------------------------------------------------------
    # 4. Stream LLM response
    # --------------------------------------------------------

    return StreamingResponse(
        stream_llm_response(prompt),
        media_type="text/event-stream",
    )


# ============================================================
# GLOBAL ERROR HANDLER
# ============================================================

@app.exception_handler(Exception)
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """
    Prevent raw tracebacks from being returned to users.
    """

    logger.exception(
        "Unhandled error on %s",
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error."
        },
    )


# ============================================================
# SELF TEST
# ============================================================

def _self_test() -> None:
    """
    Basic API self-test.

    The test verifies:
        - /health
        - PDF validation
        - upload handling
        - /chat validation
        - retriever connection
    """

    from fastapi.testclient import TestClient

    print(
        "=" * 70
    )

    print(
        "RUNNING API SELF-TEST"
    )

    print(
        "=" * 70
    )

    client = TestClient(
        app
    )

    failures = []

    # --------------------------------------------------------
    # Helper
    # --------------------------------------------------------

    def check(
        name: str,
        condition: bool,
    ) -> None:

        if condition:

            print(
                f"PASS  {name}"
            )

        else:

            failures.append(
                name
            )

            print(
                f"FAIL  {name}"
            )

    # --------------------------------------------------------
    # Health
    # --------------------------------------------------------

    response = client.get(
        "/health"
    )

    check(
        "health returns 200",
        response.status_code == 200,
    )

    check(
        "health status is ok",
        response.json().get(
            "status"
        ) == "ok",
    )

    # --------------------------------------------------------
    # Reject non-PDF
    # --------------------------------------------------------

    response = client.post(
        "/upload",
        files={
            "file": (
                "notes.txt",
                b"hello",
                "text/plain",
            )
        },
    )

    check(
        "upload rejects non-PDF",
        response.status_code == 400,
    )

    # --------------------------------------------------------
    # Reject empty PDF
    # --------------------------------------------------------

    response = client.post(
        "/upload",
        files={
            "file": (
                "empty.pdf",
                b"",
                "application/pdf",
            )
        },
    )

    check(
        "upload rejects empty PDF",
        response.status_code == 400,
    )

    # --------------------------------------------------------
    # Accept PDF
    # --------------------------------------------------------

    response = client.post(
        "/upload",
        files={
            "file": (
                "test.pdf",
                b"%PDF-1.4 test",
                "application/pdf",
            )
        },
    )

    check(
        "upload accepts PDF",
        response.status_code == 200,
    )

    check(
        "upload returns doc_id",
        bool(
            response.json().get(
                "doc_id"
            )
        ),
    )

    # --------------------------------------------------------
    # Empty chat message
    # --------------------------------------------------------

    response = client.post(
        "/chat",
        json={
            "session_id": "test-session",
            "message": "",
        },
    )

    check(
        "chat rejects empty message",
        response.status_code == 422,
    )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print()

    if failures:

        print(
            f"{len(failures)} test(s) failed:"
        )

        for failure in failures:

            print(
                f"  - {failure}"
            )

        raise SystemExit(1)

    print(
        "All API framework tests passed."
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    _self_test()