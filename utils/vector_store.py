import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

def save_vector_store(
    embeddings: np.ndarray,
    chunks: List[Dict[str, Any]],
    output_folder: Path,
    index_name: str = "faiss_index.bin",
    metadata_name: str = "metadata.json",
    normalize: bool = True
) -> None:
    """
    Saves the computed embeddings to a FAISS index and the corresponding metadata to a JSON file.
    Uses IndexFlatIP (Inner Product) which calculates Cosine Similarity when vectors are normalized.

    Args:
        embeddings: NumPy array of shape (num_chunks, dimension) containing chunk embeddings.
        chunks: List of dictionaries representing the original chunks and metadata.
        output_folder: Directory path where output files will be saved.
        index_name: Filename of the FAISS index binary.
        metadata_name: Filename of the metadata JSON.
        normalize: If True, L2 normalizes the embeddings before adding them to FAISS.
    """
    # 1. Validation
    if len(chunks) != embeddings.shape[0]:
        logger.error(
            f"Dimension mismatch: number of chunks ({len(chunks)}) "
            f"does not match number of embeddings ({embeddings.shape[0]})."
        )
        raise ValueError("The number of chunks and embeddings must be equal.")

    if len(chunks) == 0:
        logger.warning("No data to save in the vector store.")
        return

    # Create folder if it doesn't exist
    try:
        output_folder.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission denied creating vector store folder '{output_folder}': {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to create output folder: {e}")
        raise

    # 2. Normalize embeddings for Cosine Similarity (L2 normalization + Inner Product)
    working_embeddings = embeddings.copy()
    if normalize:
        logger.info("Normalizing embeddings for Inner Product (Cosine Similarity) FAISS index...")
        faiss.normalize_L2(working_embeddings)

    # 3. Create FAISS index
    dimension = working_embeddings.shape[1]
    logger.info(f"Creating FAISS IndexFlatIP with dimension {dimension}...")
    try:
        # IndexFlatIP is used for Inner Product / Cosine Similarity (on normalized vectors)
        index = faiss.IndexFlatIP(dimension)
        index.add(working_embeddings)
    except Exception as e:
        logger.error(f"Failed to create or populate FAISS index: {e}")
        raise

    # 4. Prepare Metadata list with ID mapping
    metadata_list: List[Dict[str, Any]] = []
    for idx, chunk in enumerate(chunks):
        metadata_list.append({
            "id": idx,  # Direct alignment to FAISS index ID
            "candidate_name": chunk["candidate_name"],
            "resume_file": chunk["resume_file"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"]
        })

    # 5. Save FAISS index binary
    index_path = output_folder / index_name
    logger.info(f"Saving FAISS index to {index_path}...")
    try:
        faiss.write_index(index, str(index_path))
    except Exception as e:
        logger.error(f"Failed to save FAISS index to disk: {e}")
        raise

    # 6. Save Metadata JSON
    metadata_path = output_folder / metadata_name
    logger.info(f"Saving metadata mapping to {metadata_path}...")
    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata_list, f, indent=4, ensure_ascii=False)
    except Exception as e:
        logger.error(f"Failed to save metadata JSON file to disk: {e}")
        raise

    logger.info("Vector database and metadata saved successfully.")
## why we can take  fassi database