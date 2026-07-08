import logging
import numpy as np
from typing import List
from utils.embedding_model import EmbeddingEngine

logger = logging.getLogger(__name__)

def embed_query(
    query: str,
    model_name: str = "BAAI/bge-small-en-v1.5",
    normalize: bool = True
) -> np.ndarray:
    """
    Converts a single user question/query into a normalized 1D embedding vector.

    Args:
        query: User question string.
        model_name: HuggingFace model path of the embedding model to load.
        normalize: Whether to L2 normalize the embedding (required for Cosine Similarity).

    Returns:
        A 1D float32 NumPy array representing the query embedding.
    """
    if not query or not query.strip():
        logger.error("Empty query passed to query embedder.")
        raise ValueError("Query string cannot be empty.")

    logger.info("Encoding query...")
    engine = EmbeddingEngine(model_name=model_name)
    
    # generate_embeddings returns a 2D array [num_queries, dimension]
    embeddings = engine.generate_embeddings([query], batch_size=1, normalize=normalize)
    
    return embeddings[0]

def embed_queries(
    queries: List[str],
    model_name: str = "BAAI/bge-small-en-v1.5",
    batch_size: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """
    Converts a list of query strings into a 2D matrix of embeddings (batch support).

    Args:
        queries: List of query strings.
        model_name: HuggingFace model path of the embedding model to load.
        batch_size: Batch size for encoding.
        normalize: Whether to L2 normalize the embeddings.

    Returns:
        A 2D float32 NumPy array of shape (len(queries), dimension).
    """
    if not queries:
        logger.warning("Empty list of queries passed to batch query embedder.")
        engine = EmbeddingEngine(model_name=model_name)
        dimension = engine.get_embedding_dimension()
        return np.empty((0, dimension), dtype=np.float32)

    logger.info(f"Encoding {len(queries)} queries in batch...")
    engine = EmbeddingEngine(model_name=model_name)
    
    return engine.generate_embeddings(queries, batch_size=batch_size, normalize=normalize)
