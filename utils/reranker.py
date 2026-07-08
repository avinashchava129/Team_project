import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class RerankerEngine:
    """
    Singleton re-ranker engine using a Hugging Face Cross-Encoder.
    If the model cannot be loaded or initialized (e.g., due to connection timeouts
    or missing PyTorch configurations), it fails silently and bypasses re-ranking.
    """
    _instance = None

    def __new__(cls, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if cls._instance is None:
            cls._instance = super(RerankerEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if self._initialized:
            return
        
        self.model_name = model_name
        self.model = None
        try:
            logger.info(f"Loading CrossEncoder re-ranker model '{model_name}'...")
            self.model = CrossEncoder(model_name)
            self._initialized = True
            logger.info("CrossEncoder re-ranker loaded successfully.")
        except Exception as e:
            logger.warning(
                f"CrossEncoder re-ranker '{model_name}' could not be initialized: {e}. "
                "Re-ranking will be bypassed gracefully."
            )
            # Retain self.model = None so query execution will bypass re-ranking

    def rerank_chunks(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        keep_top: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Computes re-ranking similarity scores for chunk-query pairs.
        
        Args:
            query: The user question.
            chunks: List of retrieved chunks.
            keep_top: Number of top chunks to return.

        Returns:
            Re-ranked and sliced list of chunks.
        """
        if not chunks:
            return []

        # If re-ranker model was not loaded, slice original FAISS results and return
        if self.model is None:
            logger.info(f"Re-ranking bypassed. Returning top {keep_top} chunks from FAISS.")
            return chunks[:keep_top]

        logger.info("Re-ranking chunks...")
        try:
            # Construct pairs: (query, text)
            pairs = [[query, chunk["text"]] for chunk in chunks]
            
            # Generate cross-encoder similarity scores
            scores = self.model.predict(pairs)
            
            # Store scores on each chunk dict
            for idx, score in enumerate(scores):
                chunks[idx]["rerank_score"] = float(score)

            # Sort descending by re-rank score
            reranked_chunks = sorted(chunks, key=lambda x: x["rerank_score"], reverse=True)
            
            result = reranked_chunks[:keep_top]
            logger.info(f"Re-ranked successfully. Top {len(result)} chunks selected.")
            return result

        except Exception as e:
            logger.error(f"Error during re-ranking process: {e}. Returning original FAISS ordering.")
            return chunks[:keep_top]
## why we can take this model only just explain