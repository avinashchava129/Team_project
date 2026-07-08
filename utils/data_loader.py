import json
import logging
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def load_chunks_from_json(file_path: Path) -> List[Dict[str, Any]]:
    """
    Loads and validates the text chunks from a JSON file.

    Args:
        file_path: Path to the input JSON file (typically chunks.json).

    Returns:
        A list of validated dictionaries, each representing a chunk with its metadata.
        Returns an empty list if the file is missing, empty, or invalid.
    """
    if not file_path.exists():
        logger.error(f"Input file not found: {file_path}")
        return []

    if not file_path.is_file():
        logger.error(f"Provided path is not a file: {file_path}")
        return []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Failed to parse JSON in '{file_path}': {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error reading '{file_path}': {e}")
        return []

    if not isinstance(data, list):
        logger.error(f"Invalid format in '{file_path}': expected a list of chunks, got {type(data).__name__}")
        return []

    required_keys = {"candidate_name", "resume_file", "chunk_id", "total_chunks", "text"}
    valid_chunks: List[Dict[str, Any]] = []

    for index, item in enumerate(data):
        if not isinstance(item, dict):
            logger.warning(f"Skipping index {index}: expected a JSON object, got {type(item).__name__}")
            continue

        # Check for missing keys
        missing_keys = required_keys - item.keys()
        if missing_keys:
            logger.warning(f"Skipping index {index}: missing required field(s): {', '.join(missing_keys)}")
            continue

        # Check for empty text field
        text_content = item.get("text")
        if not isinstance(text_content, str) or not text_content.strip():
            logger.warning(f"Skipping index {index} (candidate: {item.get('candidate_name')}): 'text' field is empty or not a string.")
            continue

        # Optionally validate type of chunk_id and total_chunks
        try:
            item["chunk_id"] = int(item["chunk_id"])
            item["total_chunks"] = int(item["total_chunks"])
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping index {index}: invalid non-integer chunk indices: {e}")
            continue

        valid_chunks.append(item)

    logger.info(f"Successfully loaded and validated {len(valid_chunks)} chunks out of {len(data)} total entries.")
    return valid_chunks
"loading"