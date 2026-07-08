import logging
from typing import List
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

def generate_chunks(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    separators: List[str] = None
) -> List[str]:
    """
    Returns the entire cleaned text of the resume as a single chunk.
    This fulfills the requirement of '1 resume = 1 chunk'.
    """
    if not text or not text.strip():
        logger.warning("Empty text passed to chunk generator.")
        return []

    return [text.strip()]
##types of chunks  should be  explain guys