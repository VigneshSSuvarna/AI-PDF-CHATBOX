from langchain_core.prompts import PromptTemplate

def get_strict_rag_prompt(persona: str = "professional"):
    """
    Returns an advanced strict LangChain PromptTemplate supporting 
    dynamic personas, conversation history, and strict anti-hallucination rules.
    """
    
    # Feature 1: Dynamic Persona / Tone Selection
    tone_instructions = {
        "professional": "You are a helpful, professional, and precise AI PDF Chatbot.",
        "concise": "You are a direct and concise AI PDF Chatbot. Provide brief, straightforward answers.",
        "expert": "You are an expert domain analyst AI. Provide thorough, structured breakdowns based exclusively on the context."
    }
    selected_persona = tone_instructions.get(persona, tone_instructions["professional"])

    template_string = f"""{selected_persona}
Use the following conversation history and retrieved context to answer the user's question. 
If the answer is not contained within the context, you must explicitly say 'I don't know based on the provided documents.' 
Do not make up or hallucinate information. Always cite your source file, page number, and chunk index when referencing facts.

Conversation History:
{{chat_history}}

Retrieved Context: 
{{retrieved_context}}

User Question: 
{{user_question}}

Answer:"""

    return PromptTemplate(
        input_variables=["chat_history", "retrieved_context", "user_question"],
        template=template_string
    )


def combine_retrieved_context(retrieved_docs):
    """
    Takes ChromaDB document chunks, cleans excessive whitespace, adds 
    sequential chunk indices, and formats metadata for precise LLM citations.
    """
    formatted_chunks = []
    
    for idx, doc in enumerate(retrieved_docs, start=1):
        # Extract metadata saved during ingestion
        source_file = doc.metadata.get("source", "Unknown Document")
        page_num = doc.metadata.get("page", "Unknown Page")
        
        # Feature 3: Clean up messy whitespace/newlines from raw PDF extraction
        cleaned_content = " ".join(doc.page_content.split())
        
        # Feature 2: Format with clear sequential Chunk IDs and metadata markers
        formatted_chunk = f"[Chunk {idx} | Source: {source_file} | Page: {page_num}]\n{cleaned_content}"
        formatted_chunks.append(formatted_chunk)
        
    # Join all chunks together with clear visual separators
    return "\n\n---\n\n".join(formatted_chunks)