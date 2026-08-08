import os
from typing import Any, Dict, List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma


# ============================================================
# CONFIGURATION
# ============================================================

CHROMA_DB_PATH = "chroma_db"

COLLECTION_NAME = "capstone_knowledge_base"

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

DEFAULT_TOP_K = 5


# ============================================================
# RETRIEVER
# ============================================================

class PDFRetriever:
    """
    Semantic retriever for the AI PDF Chatbox.

    It converts a user query into an embedding and searches
    the ChromaDB vector database for the most relevant chunks.
    """

    def __init__(
        self,
        db_path: str = CHROMA_DB_PATH,
        collection_name: str = COLLECTION_NAME,
        embedding_model: str = EMBEDDING_MODEL,
    ):

        self.db_path = db_path
        self.collection_name = collection_name
        self.embedding_model = embedding_model

        # ----------------------------------------------------
        # Check ChromaDB
        # ----------------------------------------------------

        if not os.path.exists(self.db_path):
            raise FileNotFoundError(
                f"ChromaDB directory not found: {self.db_path}\n"
                "Please run vector_store.py first."
            )

        # ----------------------------------------------------
        # Load embedding model
        # ----------------------------------------------------

        print(
            f"Loading embedding model: "
            f"{self.embedding_model}"
        )

        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={
                "device": "cpu"
            },
            encode_kwargs={
                "normalize_embeddings": True
            }
        )

        # ----------------------------------------------------
        # Connect to ChromaDB
        # ----------------------------------------------------

        print("Connecting to ChromaDB...")

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.db_path,
        )

        print("Connected successfully.")

        # ----------------------------------------------------
        # Display database information
        # ----------------------------------------------------

        try:

            count = self.vector_store._collection.count()

            print(
                f"Total chunks in database: {count}"
            )

        except Exception:

            print(
                "Could not determine number of chunks."
            )


    # ========================================================
    # EMBED USER QUERY
    # ========================================================

    def embed_query(self, query: str) -> List[float]:
        """
        Convert the user's question into an embedding vector.

        Example:

            "What is artificial intelligence?"

        becomes a numerical vector.
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

        vector = self.embeddings.embed_query(query)

        return vector


    # ========================================================
    # BASIC RETRIEVAL
    # ========================================================

    def retrieve(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ):
        """
        Retrieve the most relevant document chunks.

        Parameters
        ----------
        query : str
            User's question.

        top_k : int
            Number of chunks to retrieve.

        Returns
        -------
        List
            List of LangChain Document objects.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # ----------------------------------------------------
        # Convert query into vector
        # ----------------------------------------------------

        query_vector = self.embed_query(query)

        # ----------------------------------------------------
        # Similarity search
        # ----------------------------------------------------

        documents = (
            self.vector_store
            .similarity_search_by_vector(
                embedding=query_vector,
                k=top_k
            )
        )

        return documents


    # ========================================================
    # RETRIEVAL WITH SCORES
    # ========================================================

    def retrieve_with_scores(
        self,
        query: str,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents along with their ChromaDB
        distance scores.

        Lower distance generally means the vectors are
        more similar.
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        # ----------------------------------------------------
        # Generate query embedding
        # ----------------------------------------------------

        query_vector = self.embed_query(query)

        # ----------------------------------------------------
        # Access Chroma collection
        # ----------------------------------------------------

        collection = self.vector_store._collection

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        retrieved_results = []

        for i in range(len(documents)):

            retrieved_results.append(
                {
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i]
                }
            )

        return retrieved_results


    # ========================================================
    # METADATA FILTERING
    # ========================================================

    def retrieve_with_filter(
        self,
        query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        top_k: int = DEFAULT_TOP_K
    ) -> List[Dict[str, Any]]:
        """
        Perform semantic search with an optional metadata filter.

        Example:

            metadata_filter = {
                "source": "example.pdf"
            }
        """

        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        query_vector = self.embed_query(query)

        collection = self.vector_store._collection

        search_parameters = {
            "query_embeddings": [query_vector],
            "n_results": top_k,
            "include": [
                "documents",
                "metadatas",
                "distances"
            ]
        }

        # ----------------------------------------------------
        # Add metadata filter
        # ----------------------------------------------------

        if metadata_filter:

            search_parameters["where"] = metadata_filter

        # ----------------------------------------------------
        # Search
        # ----------------------------------------------------

        results = collection.query(
            **search_parameters
        )

        documents = results.get(
            "documents",
            [[]]
        )[0]

        metadatas = results.get(
            "metadatas",
            [[]]
        )[0]

        distances = results.get(
            "distances",
            [[]]
        )[0]

        retrieved_results = []

        for i in range(len(documents)):

            retrieved_results.append(
                {
                    "content": documents[i],
                    "metadata": metadatas[i],
                    "distance": distances[i]
                }
            )

        return retrieved_results


    # ========================================================
    # DISPLAY NORMAL DOCUMENT RESULTS
    # ========================================================

    def display_documents(
        self,
        documents
    ):
        """
        Display LangChain Document objects.
        """

        print("\n")
        print("=" * 80)
        print("RETRIEVED RESULTS")
        print("=" * 80)

        if not documents:

            print("No relevant documents found.")

            return

        for index, document in enumerate(
            documents,
            start=1
        ):

            print("\n")
            print("-" * 80)
            print(f"RESULT {index}")
            print("-" * 80)

            print("\nCONTENT:")
            print(document.page_content)

            print("\nMETADATA:")

            if document.metadata:

                for key, value in (
                    document.metadata.items()
                ):

                    print(
                        f"  {key}: {value}"
                    )

            else:

                print("  No metadata available.")

        print("\n")
        print("=" * 80)


    # ========================================================
    # DISPLAY SCORED RESULTS
    # ========================================================

    def display_scored_results(
        self,
        results
    ):
        """
        Display results with similarity/distance scores.
        """

        print("\n")
        print("=" * 80)
        print("TOP RETRIEVED CHUNKS")
        print("=" * 80)

        if not results:

            print("No relevant documents found.")

            return

        for index, result in enumerate(
            results,
            start=1
        ):

            print("\n")
            print("-" * 80)
            print(f"RESULT {index}")
            print("-" * 80)

            print(
                f"\nDistance: "
                f"{result['distance']:.6f}"
            )

            print("\nCONTENT:")

            print(
                result["content"]
            )

            print("\nMETADATA:")

            metadata = result.get(
                "metadata",
                {}
            )

            if metadata:

                for key, value in metadata.items():

                    print(
                        f"  {key}: {value}"
                    )

            else:

                print(
                    "  No metadata available."
                )

        print("\n")
        print("=" * 80)


# ============================================================
# TERMINAL SEARCH
# ============================================================

def terminal_search():

    print("\n")
    print("=" * 80)
    print("AI PDF CHATBOX")
    print("CORE RETRIEVER")
    print("=" * 80)

    try:

        retriever = PDFRetriever()

    except Exception as error:

        print("\nERROR:")
        print(error)

        return

    print("\nRetriever is ready.")

    print(
        "Enter a question to search the knowledge base."
    )

    print(
        "Type 'exit' to quit."
    )

    # --------------------------------------------------------
    # Search loop
    # --------------------------------------------------------

    while True:

        try:

            query = input(
                "\nQuestion: "
            ).strip()

            # ------------------------------------------------
            # Empty query
            # ------------------------------------------------

            if not query:

                print(
                    "Please enter a question."
                )

                continue

            # ------------------------------------------------
            # Exit
            # ------------------------------------------------

            if query.lower() in [
                "exit",
                "quit",
                "q"
            ]:

                print(
                    "\nExiting retriever..."
                )

                break

            # ------------------------------------------------
            # Perform retrieval
            # ------------------------------------------------

            print(
                "\nSearching..."
            )

            results = retriever.retrieve_with_scores(
                query=query,
                top_k=5
            )

            # ------------------------------------------------
            # Display results
            # ------------------------------------------------

            retriever.display_scored_results(
                results
            )

        except KeyboardInterrupt:

            print(
                "\n\nProgram stopped."
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

    terminal_search()