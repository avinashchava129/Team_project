import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def build_context_block(chunks: List[Dict[str, Any]]) -> str:
    """
    Formats a list of retrieved resume chunks into a single readable context string.

    Args:
        chunks: List of retrieved text chunks with metadata.

    Returns:
        A formatted string containing candidate names, source filenames, and text content.
    """
    if not chunks:
        return "[No relevant resume sections were found in the database.]"

    context_parts = []
    for index, chunk in enumerate(chunks):
        candidate = chunk.get("candidate", "Unknown Candidate")
        filename = chunk.get("resume_file", "Unknown File")
        text = chunk.get("text", "").strip()
        
        # Structure the chunk with source identifiers for LLM reading clarity
        part = f"--- Document {index + 1} ---\n"
        part += f"Candidate: {candidate}\n"
        part += f"Source File: {filename}\n"
        part += f"Content:\n{text}\n"
        context_parts.append(part)

    return "\n".join(context_parts)

def build_rag_prompt(retrieved_context: str, question: str) -> str:
    """
    Fills the professional RAG prompt template with context and the user's question.

    Args:
        retrieved_context: Formatted context block.
        question: User query string.

    Returns:
        The complete prompt string ready for LLM input.
    """
    logger.info("Building prompt...")
    
    # Strictly aligned with the requested RAG template
    prompt = f"""You are an AI Resume Assistant.

Use ONLY the information provided in the context.

If the answer cannot be found in the context, respond exactly:

"I could not find that information in the uploaded resumes."

Do NOT guess.

Do NOT fabricate information.

Context:

{retrieved_context}

Question:

{question}

Answer:"""
    return prompt
