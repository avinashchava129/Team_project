import json
import logging
import numpy as np
import faiss
from pathlib import Path
from typing import List, Dict, Any, Tuple

logger = logging.getLogger(__name__)

def load_vector_store(
    index_path: Path,
    metadata_path: Path
) -> Tuple[faiss.Index, List[Dict[str, Any]]]:
    """
    Loads the FAISS index and the corresponding metadata from disk.

    Args:
        index_path: Path to the FAISS index file (e.g., faiss_index.bin).
        metadata_path: Path to the metadata JSON file.

    Returns:
        A tuple of (faiss_index, metadata_list).
    """
    logger.info("Loading FAISS index...")
    if not index_path.exists():
        logger.error(f"FAISS index file not found: {index_path}")
        raise FileNotFoundError(f"FAISS index file not found: {index_path}")

    try:
        index = faiss.read_index(str(index_path))
    except Exception as e:
        logger.error(f"Failed to read FAISS index from '{index_path}': {e}")
        raise

    logger.info("Loading metadata...")
    if not metadata_path.exists():
        logger.error(f"Metadata file not found: {metadata_path}")
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    except Exception as e:
        logger.error(f"Failed to read metadata file from '{metadata_path}': {e}")
        raise

    return index, metadata

def get_matched_candidates(query_text: str, metadata: List[Dict[str, Any]]) -> List[str]:
    """
    Scans the query text for keywords belonging to candidate names or resume filenames.
    Returns a list of matching resume filenames.
    """
    import re
    from pathlib import Path

    if not query_text or not metadata:
        return []

    query_lower = query_text.lower()
    query_tokens = set(re.findall(r'\b\w+\b', query_lower))
    filtered_filenames = []
    
    # Map unique filenames to candidate names
    candidate_map = {}
    for meta in metadata:
        fname = meta.get("resume_file", "")
        cname = meta.get("candidate_name", "")
        if fname and cname:
            candidate_map[fname] = cname
            
    for filename, candidate_name in candidate_map.items():
        # Clean and tokenize words from candidate name and filename stem
        name_words = re.findall(r'\b\w+\b', candidate_name.lower())
        file_words = re.findall(r'\b\w+\b', Path(filename).stem.lower())
        
        # Stopwords or generic words to ignore
        ignored_words = {
            'and', 'the', 'for', 'with', 'pdf', 'resume', 'cv', 'profile', 
            'updated', 'main', 'responses', 'file', 'sample', 'groundtruth',
            'reddy', 'kumar', 'singh', 'sharma', 'prasad', 'chava', 'muddada', 
            'gorintala', 'nadimpalli', 'vellanki', 'malle', 'btech', 'csd',
            'project', 'projects'
        }
        
        # Keywords must be at least 3 characters and not ignored
        keywords = set(name_words + file_words) - ignored_words
        keywords = {w for w in keywords if len(w) >= 3}
        
        for word in keywords:
            if word in query_tokens:
                logger.info(f"Metadata filter match: keyword '{word}' in query matched candidate file '{filename}'")
                filtered_filenames.append(filename)
                break
                
    return filtered_filenames

def retrieve_chunks(
    query_vector: np.ndarray,
    index: faiss.Index,
    metadata: List[Dict[str, Any]],
    top_k: int = 5,
    query_text: str = ""
) -> List[Dict[str, Any]]:
    """
    Performs similarity search using the FAISS index and returns Top-K chunks.
    Optionally applies a metadata candidate-name filter if the query specifies a candidate.

    Args:
        query_vector: 1D or 2D query embedding.
        index: Loaded FAISS index.
        metadata: List of metadata dictionaries.
        top_k: Number of chunks to retrieve.
        query_text: User question text to extract candidate keywords for filtering.

    Returns:
        A list of dictionary objects representing retrieved chunks.
    """
    if index is None or not metadata:
        logger.error("FAISS index or metadata is not initialized.")
        return []

    # Ensure query vector is 2D float32
    if len(query_vector.shape) == 1:
        query_vector = np.expand_dims(query_vector, axis=0)
    query_vector = query_vector.astype(np.float32)

    # 1. Candidate Name Filtering Heuristic
    filtered_filenames = get_matched_candidates(query_text, metadata)

    # If filtering matches files, expand search size to scan more candidates
    actual_top_k = top_k
    search_k = top_k
    if filtered_filenames:
        search_k = min(100, len(metadata))
        actual_top_k = 15  # Retrieve more candidate chunks to let the reranker score them

    logger.info(f"Retrieving top {search_k} chunks...")
    try:
        scores, indices = index.search(query_vector, search_k)
    except Exception as e:
        logger.error(f"FAISS index search failed: {e}")
        return []

    retrieved_chunks = []
    
    # 2. Extract and filter search results
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
            
        if idx >= len(metadata):
            continue

        meta = metadata[idx]
        filename = meta.get("resume_file", "")
        
        # If filtering is active, skip chunks that belong to other candidates
        if filtered_filenames and filename not in filtered_filenames:
            continue

        retrieved_chunks.append({
            "score": float(score),
            "candidate": meta.get("candidate_name", ""),
            "candidate_name": meta.get("candidate_name", ""),
            "resume_file": filename,
            "chunk_id": meta.get("chunk_id", -1),
            "text": meta.get("text", "")
        })
        
        # Break once we've collected actual_top_k matched chunks
        if len(retrieved_chunks) >= actual_top_k:
            break

    # If filtering yielded 0 results (e.g. strict fallback), do unfiltered search
    if not retrieved_chunks and filtered_filenames:
        logger.warning("Metadata filtering yielded 0 results. Falling back to unfiltered search.")
        return retrieve_chunks(query_vector, index, metadata, top_k, query_text="")

    logger.info(f"Retrieved {len(retrieved_chunks)} chunks successfully (filtered={bool(filtered_filenames)}).")
    return retrieved_chunks

