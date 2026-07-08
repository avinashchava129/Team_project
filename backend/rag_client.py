#this file belongs to the RAG client module that wraps Member 3's RAGPipeline and provides a clean API for the Streamlit frontend to query the index.
import logging
from typing import Dict, Any
from rag_engine import RAGPipeline
import config

logger = logging.getLogger(__name__)

class RAGClient:
    """
    Integration client wrapping Member 3's RAGPipeline.
    Provides a clean API for the Streamlit frontend to query the index.
    """
    def __init__(self) -> None:
        self.pipeline = RAGPipeline()
        self.initialized = False

    def query(self, question: str, model_name: str = None) -> Dict[str, Any]:
        """
        Sends the user question to Member 3's RAG pipeline.

        Args:
            question: Natural language question.
            model_name: Name of the LLM model to use (Ollama tags or Gemini).

        Returns:
            Structured dictionary response:
            {
                "question": str,
                "answer": str,
                "retrieved_chunks": list[dict],
                "sources": list[str],
                "similarity_scores": list[float]
            }
        """
        # Dynamic model switching support
        if model_name:
            config.MODEL_NAME = model_name
            # If the model is a gemini model name, set USE_GEMINI to True, else False
            if "gemini" in model_name.lower():
                config.USE_GEMINI = True
                config.GEMINI_MODEL = model_name
            else:
                config.USE_GEMINI = False
                config.MODEL_NAME = model_name

        if not self.initialized:
            logger.info("Initializing RAG Client pipeline...")
            success = self.pipeline.initialize()
            if not success:
                logger.error("RAG pipeline failed to initialize.")
                return {
                    "question": question,
                    "answer": "Error: RAG pipeline initialization failed. Ensure vector database is built and model is available.",
                    "retrieved_chunks": [],
                    "sources": [],
                    "similarity_scores": []
                }
            self.initialized = True

        logger.info(f"Querying RAG Engine with question: '{question}' using model '{model_name}'")
        try:
            return self.pipeline.query(question)
        except Exception as e:
            logger.error(f"Failed to query RAG Engine: {e}")
            return {
                "question": question,
                "answer": f"Error: Unexpected failure during query execution: {e}",
                "retrieved_chunks": [],
                "sources": [],
                "similarity_scores": []
            }

    def reset_index(self) -> None:
        """
        Resets the active pipeline state.
        Forces the RAG engine to reload index and metadata files from disk on the next query.
        """
        logger.info("Resetting RAG Client index context.")
        self.pipeline.initialized = False
        self.initialized = False
