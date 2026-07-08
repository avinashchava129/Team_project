import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """
    Singleton class to manage the SentenceTransformer model loading and embedding generation.
    Ensures that the model is loaded only once in memory.
    """
    _instance = None

    def __new__(cls, model_name: str = "BAAI/bge-small-en-v1.5"):
        if cls._instance is None:
            cls._instance = super(EmbeddingEngine, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        if self._initialized:
            return
        
        logger.info(f"Loading embedding model '{model_name}'...")
        try:
            # Load the SentenceTransformer model
            self.model = SentenceTransformer(model_name)
            self.model_name = model_name
            self._initialized = True
            logger.info("Embedding model loaded successfully.")
        except Exception as e:
            logger.critical(f"Failed to load embedding model '{model_name}': {e}")
            raise RuntimeError(f"Could not load embedding model: {e}") from e

    def get_embedding_dimension(self) -> int:
        """
        Returns the output vector dimension of the loaded embedding model.
        """
        try:
            return self.model.get_sentence_embedding_dimension()
        except Exception:
            # Fallback for standard models
            dummy_emb = self.model.encode(["test"])
            return dummy_emb.shape[1]

    def generate_embeddings(
        self,
        texts: List[str],
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Generates embeddings for a list of texts using batch processing.

        Args:
            texts: List of text strings to embed.
            batch_size: Batch size used for encoding.
            normalize: If True, L2 normalizes the embeddings (required for Cosine Similarity).

        Returns:
            A float32 NumPy array of shape (num_texts, embedding_dimension).
        """
        if not texts:
            logger.warning("Empty list of texts passed for embedding generation.")
            dimension = self.get_embedding_dimension()
            return np.empty((0, dimension), dtype=np.float32)

        logger.info(f"Generating embeddings for {len(texts)} chunks...")
        try:
            # Perform batch encoding
            embeddings = self.model.encode(
                sentences=texts,
                batch_size=batch_size,
                show_progress_bar=True,
                normalize_embeddings=normalize,
                convert_to_numpy=True
            )
            # Ensure float32 dtype for FAISS compatibility
            return embeddings.astype(np.float32)
        except Exception as e:
            logger.error(f"Error during batch embedding generation: {e}")
            raise
