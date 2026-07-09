import re
from pathlib import Path
from typing import Dict, Any

def extract_candidate_name(filename: str) -> str:
    """
    Extracts and cleans the candidate's name from the resume filename.
    Considers the parts split by ' - ' which commonly contain the clean candidate name.
    
    Example:
    - 'AVINASH GROUNDTRUTH RESUME - Avinash Chava.pdf' -> 'Avinash Chava'
    - 'Abduljaha_Parchuru_AI_Resume - parchuru abduljaha.pdf' -> 'Parchuru Abduljaha'
    """
    stem = Path(filename).stem
    
    if " - " in stem:
        parts = stem.split(" - ")
        name_part = parts[1].strip()
        # If the second  part is too short or generic, fallback to the first part
        if name_part and len(name_part) > 2 and name_part.lower() not in ["resume", "cv", "main"]:
            name = name_part
        else:
            name = parts[0].strip()
    else:
        name = stem
        
    # Clean underscores and hyphens, replace with spaces
    name = name.replace("_", " ").replace("-", " ")
    
    # Normalize whitespaces and clean
    name = re.sub(r"\s+", " ", name)
    
    # Format to Title Case
    return name.title().strip()

def create_chunk_metadata(
    candidate_name: str,
    resume_file: str,
    chunk_id: int,
    total_chunks: int,
    text: str
) -> Dict[str, Any]:
    """
    Builds the structured dictionary containing a text chunk and its metadata.

    Args:
        candidate_name: Extracted name of the candidate.
        resume_file: Original filename of the resume PDF.
        chunk_id: The 0-based index of this chunk.
        total_chunks: Total number of chunks generated for this resume.
        text: The text content of this chunk.

    Returns:
        A dictionary representation of the chunk and its metadata.
    """
    return {
        "candidate_name": candidate_name,
        "resume_file": resume_file,
        "chunk_id": chunk_id,
        "total_chunks": total_chunks,
        "text": text
    }
