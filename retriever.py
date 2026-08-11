"""
retriever.py
============

Core Retriever Logic for AI-PDF-CHATBOX

Purpose:
    1. Accept a user's question.
    2. Convert the question into an embedding.
    3. Search the ChromaDB vector database.
    4. Return the most relevant document chunks.

This module is designed to be easily imported into
the Week 3 FastAPI + LLM backend.
"""

import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DB_PATH = "chroma_db"

COLLECTION_NAME = "capstone_knowledge_base"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 5


# ============================================================
# RETRIEVER CLASS
# ============================================================

class Retriever:
    """
    Semantic retriever for the AI-PDF-CHATBOX.

    The same embedding model used to create the ChromaDB
    vectors must be used to embed the user's query.
    """

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        model_name: str = EMBEDDING_MODEL_NAME,
    ) -> None:

        self.db_path = db_path
        self.collection_name = collection_name
        self.model_name = model_name

        # ----------------------------------------------------
        # Check whether ChromaDB exists
        # ----------------------------------------------------

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"ChromaDB directory '{self.db_path}' was not found.\n"
                "Please run vector_store.py first."
            )

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        print(
            f"Loading embedding model: {self.model_name}"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.model_name,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            },
        )

        # ----------------------------------------------------
        # Connect to existing ChromaDB
        # ----------------------------------------------------

        print("Connecting to ChromaDB...")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )

        print("Connected to ChromaDB successfully.")

        # ----------------------------------------------------
        # Display number of stored chunks
        # ----------------------------------------------------

        try:
            count = self.vector_store._collection.count()

            print(
                f"Chunks available for search: {count}"
            )

        except Exception:
            pass


    # ========================================================
    # EMBED QUERY
    # ========================================================

    def embed_query(self, query: str) -> List[float]:
        """
        Convert a user's question into an embedding vector.

        Example:

            "What is artificial intelligence?"

        becomes a numerical vector.
        """

        query = self._validate_query(query)

        return self.embeddings.embed_query(query)


    # ========================================================
    # MAIN SEARCH METHOD
    # ========================================================

    def search(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ):
        """
        Retrieve the most relevant document chunks.

        Parameters
        ----------
        query:
            User's natural-language question.

        top_k:
            Number of chunks to retrieve.

        metadata_filter:
            Optional ChromaDB metadata filter.

        Returns
        -------
        List[Document]
            Relevant LangChain Document objects.
        """

        query = self._validate_query(query)

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # ----------------------------------------------------
        # Perform semantic similarity search
        # ----------------------------------------------------

        if metadata_filter:

            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
                filter=metadata_filter,
            )

        else:

            results = self.vector_store.similarity_search(
                query=query,
                k=top_k,
            )

        return results


    # ========================================================
    # SEARCH WITH SCORES
    # ========================================================

    def search_with_scores(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Any, float]]:
        """
        Retrieve relevant chunks along with their distance scores.

        Returns:

            [
                (Document, score),
                (Document, score),
                ...
            ]

        Lower distance generally means greater similarity.
        """

        query = self._validate_query(query)

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than 0."
            )

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        if metadata_filter:

            results = (
                self.vector_store
                .similarity_search_with_score(
                    query=query,
                    k=top_k,
                    filter=metadata_filter,
                )
            )

        else:

            results = (
                self.vector_store
                .similarity_search_with_score(
                    query=query,
                    k=top_k,
                )
            )

        return results


    # ========================================================
    # RETURN CONTEXT FOR LLM
    # ========================================================

    def get_context(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Retrieve relevant chunks and combine them into one
        context string.

        This method is especially useful for Week 3.

        Example:

            context = retriever.get_context(
                "What is machine learning?"
            )

            # Pass context to the LLM.
        """

        documents = self.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        if not documents:
            return ""

        context_parts = []

        for document in documents:

            context_parts.append(
                document.page_content
            )

        return "\n\n".join(context_parts)


    # ========================================================
    # RETURN RESULTS WITH METADATA
    # ========================================================

    def get_results(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return search results in a simple dictionary format.

        Useful for APIs and JSON responses.
        """

        documents = self.search(
            query=query,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

        results = []

        for rank, document in enumerate(
            documents,
            start=1,
        ):

            results.append(
                {
                    "rank": rank,
                    "content": document.page_content,
                    "metadata": document.metadata,
                }
            )

        return results


    # ========================================================
    # VALIDATE QUERY
    # ========================================================

    @staticmethod
    def _validate_query(query: str) -> str:
        """
        Validate and clean the user's query.
        """

        if not isinstance(query, str):
            raise TypeError(
                "Query must be a string."
            )

        query = query.strip()

        if not query:
            raise ValueError(
                "Query cannot be empty."
            )

        return query


# ============================================================
# SIMPLE HELPER FUNCTION
# ============================================================

def retrieve(
    query: str,
    top_k: int = DEFAULT_TOP_K,
    metadata_filter: Optional[Dict[str, Any]] = None,
):
    """
    Simple function for retrieving documents.

    Example:

        results = retrieve(
            "What is artificial intelligence?",
            top_k=5
        )
    """

    retriever = Retriever()

    return retriever.search(
        query=query,
        top_k=top_k,
        metadata_filter=metadata_filter,
    )


# ============================================================
# TERMINAL TEST
# ============================================================

def terminal_test() -> None:
    """
    Simple terminal interface for testing the retriever.

    This is only a test interface.
    Week 3 can import Retriever directly.
    """

    print()
    print("=" * 70)
    print("AI PDF CHATBOX - CORE RETRIEVER")
    print("=" * 70)

    try:

        retriever = Retriever()

    except Exception as error:

        print()
        print("ERROR INITIALIZING RETRIEVER")
        print("-" * 70)
        print(error)
        print("-" * 70)

        return

    print()
    print("Retriever is ready.")
    print("Ask a question to search the knowledge base.")
    print("Type 'exit' to quit.")

    # --------------------------------------------------------
    # Interactive search loop
    # --------------------------------------------------------

    while True:

        try:

            query = input(
                "\nQuestion: "
            ).strip()

            # ------------------------------------------------
            # Exit commands
            # ------------------------------------------------

            if query.lower() in {
                "exit",
                "quit",
                "q",
            }:

                print(
                    "\nExiting retriever..."
                )

                break

            # ------------------------------------------------
            # Empty query
            # ------------------------------------------------

            if not query:

                print(
                    "Please enter a question."
                )

                continue

            # ------------------------------------------------
            # Search
            # ------------------------------------------------

            print(
                "\nSearching..."
            )

            results = retriever.search_with_scores(
                query=query,
                top_k=5,
            )

            # ------------------------------------------------
            # No results
            # ------------------------------------------------

            if not results:

                print(
                    "\nNo relevant results found."
                )

                continue

            # ------------------------------------------------
            # Display results
            # ------------------------------------------------

            print()
            print("=" * 70)
            print("TOP RETRIEVED CHUNKS")
            print("=" * 70)

            for index, (document, score) in enumerate(
                results,
                start=1,
            ):

                print()
                print("-" * 70)
                print(f"RESULT {index}")
                print("-" * 70)

                print(
                    f"Distance: {score:.6f}"
                )

                print()
                print("CONTENT:")
                print(
                    document.page_content
                )

                print()
                print("METADATA:")

                if document.metadata:

                    for key, value in (
                        document.metadata.items()
                    ):

                        print(
                            f"  {key}: {value}"
                        )

                else:

                    print(
                        "  No metadata available."
                    )

            print()
            print("=" * 70)

        except KeyboardInterrupt:

            print(
                "\n\nExiting retriever..."
            )

            break

        except Exception as error:

            print(
                f"\nERROR: {error}"
            )


# ============================================================
# PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    terminal_test()