from langchain_core.prompts import PromptTemplate

def get_strict_rag_prompt():
    """
    Returns an upgraded strict LangChain PromptTemplate that supports 
    conversation history (Member 3), retrieved context, and strict anti-hallucination rules.
    """
    template_string = """You are a helpful, professional, and precise AI PDF Chatbot. 
Use the following conversation history and retrieved context to answer the user's question. 
If the answer is not contained within the context, you must explicitly say 'I don't know based on the provided documents.' 
Do not make up or hallucinate information. Always cite your source file and page number using the provided metadata.

Conversation History:
{chat_history}

Retrieved Context: 
{retrieved_context}

User Question: 
{user_question}

Answer:"""

    return PromptTemplate(
        input_variables=["chat_history", "retrieved_context", "user_question"],
        template=template_string
    )


def combine_retrieved_context(retrieved_docs):
    """
    Takes the document chunks returned by ChromaDB, extracts their text 
    along with metadata (source file and page), and formats them into a single block.
    """
    formatted_chunks = []
    
    for doc in retrieved_docs:
        # Extract metadata saved during Week 1/2 ingestion
        source_file = doc.metadata.get("source", "Unknown Document")
        page_num = doc.metadata.get("page", "Unknown Page")
        content = doc.page_content
        
        # Format with clear source markers for the LLM
        formatted_chunk = f"[Source: {source_file} | Page: {page_num}]\n{content}"
        formatted_chunks.append(formatted_chunk)
        
    # Join all chunks together with clear separators
    return "\n\n---\n\n".join(formatted_chunks)