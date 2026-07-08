import torch
import sys
import logging
import subprocess
from pathlib import Path
from typing import Dict, Any

import config
from utils.query_embedder import embed_query
from utils.retriever import load_vector_store, retrieve_chunks
from utils.reranker import RerankerEngine
from utils.prompt_builder import build_context_block, build_rag_prompt
from utils.llm_client import generate_answer
from utils.response_formatter import format_rag_response, response_to_json_string

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

class RAGPipeline:
    """
    Retrieval-Augmented Generation (RAG) pipeline for resume QA.
    Retrieves semantic matches from a FAISS index, scores them with a Cross-Encoder,
    and runs a local Ollama LLM to answer the question using only retrieved context.
    """
    def __init__(self) -> None:
        self.index = None
        self.metadata = None
        self.reranker = None
        self.initialized = False

    def initialize(self) -> bool:
        """
        Loads vector store dependencies (FAISS index, metadata) and the re-ranker.
        Returns:
            True if all elements loaded successfully, False otherwise.
        """
        if self.initialized:
            return True
            
        logger.info("Loading FAISS index")
        logger.info("Loading metadata")
        try:
            self.index, self.metadata = load_vector_store(
                config.FAISS_INDEX_PATH,
                config.METADATA_PATH
            )
        except FileNotFoundError as e:
            logger.error(
                f"Required database files are missing: {e}. "
                "Please run build_vector_db.py first to compile the store."
            )
            return False
        except Exception as e:
            logger.error(f"Error loading vector store: {e}")
            return False

        logger.info("Loading embedding model")
        try:
            # Import and load embedding model once to cache it
            from utils.embedding_model import EmbeddingEngine
            EmbeddingEngine(config.EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            return False

        # Load re-ranker if enabled in config
        if config.USE_RERANKER:
            try:
                self.reranker = RerankerEngine(config.RERANKER_MODEL)
            except Exception as e:
                logger.warning(f"Re-ranker model could not be initialized: {e}. Step will be bypassed.")

        self.initialized = True
        return True

    def query(self, question: str) -> Dict[str, Any]:
        """
        Processes a single user question through the complete RAG pipeline.

        Args:
            question: The user query string.

        Returns:
            A structured dictionary containing the question, generated answer,
            retrieved context chunks, sources, and similarity scores.
        """
        if not question or not question.strip():
            logger.error("Empty query received.")
            return format_rag_response(
                question="",
                answer="Error: Query is empty.",
                chunks=[]
            )

        # Lazy initialize if needed
        if not self.initialize():
            return format_rag_response(
                question=question,
                answer="Error: RAG pipeline initialization failed. Ensure vector database is built.",
                chunks=[]
            )

        # Check if the query mentions any candidate
        from utils.retriever import get_matched_candidates
        matched_candidates = get_matched_candidates(question, self.metadata)
        if not matched_candidates:
            logger.warning("No candidate name mentioned in query.")
            return format_rag_response(
                question=question,
                answer="When you are asking a question, you should mention the person or candidate you are referring to.",
                chunks=[]
            )

        # 1. Generate query embedding
        logger.info("Encoding query")
        try:
            query_vector = embed_query(question, model_name=config.EMBEDDING_MODEL)
        except Exception as e:
            logger.error(f"Query embedding generation failed: {e}")
            return format_rag_response(
                question=question,
                answer=f"Error: Query embedding failed: {e}",
                chunks=[]
            )

        # 2. Retrieve Top-K
        logger.info("Retrieving top chunks")
        try:
            retrieved_chunks = retrieve_chunks(
                query_vector=query_vector,
                index=self.index,
                metadata=self.metadata,
                top_k=config.TOP_K,
                query_text=question
            )
        except Exception as e:
            logger.error(f"FAISS index retrieval failed: {e}")
            return format_rag_response(
                question=question,
                answer=f"Error: Context retrieval failed: {e}",
                chunks=[]
            )

        if not retrieved_chunks:
            logger.warning("No relevant chunks found in the database.")
            return format_rag_response(
                question=question,
                answer="I could not find that information in the uploaded resumes.",
                chunks=[]
            )

        # 3. Optional Re-ranking
        processed_chunks = retrieved_chunks
        if config.USE_RERANKER and self.reranker:
            logger.info("Re-ranking chunks")
            processed_chunks = self.reranker.rerank_chunks(
                query=question,
                chunks=retrieved_chunks,
                keep_top=config.RERANK_TOP_K
            )

        # 4. Build prompt
        logger.info("Building prompt")
        context_block = build_context_block(processed_chunks)
        prompt = build_rag_prompt(context_block, question)

        # 5. Call LLM (Ollama)
        logger.info("Calling LLM")
        try:
            answer = generate_answer(
                prompt=prompt,
                model_name=config.MODEL_NAME,
                host=config.OLLAMA_HOST,
                temperature=config.TEMPERATURE,
                max_tokens=config.MAX_NEW_TOKENS,
                top_p=config.TOP_P,
                timeout=config.TIMEOUT
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            answer = f"Error: LLM failed to generate a response: {e}"

        # 6. Format Response
        logger.info("Returning response")
        return format_rag_response(
            question=question,
            answer=answer,
            chunks=processed_chunks
        )

def main() -> None:
    """
    Runs the RAG Engine in an interactive command-line interface.
    """
    pipeline = RAGPipeline()

    # Self-healing database check: Build the database if files are missing
    if not config.FAISS_INDEX_PATH.exists() or not config.METADATA_PATH.exists():
        print("Vector database files not found. Auto-building vector store...")
        try:
            subprocess.run(["python", "build_vector_db.py"], check=True)
        except Exception as e:
            print(f"Error building vector database: {e}")
            sys.exit(1)

    print("Initializing RAG Pipeline (Member 3)...")
    if not pipeline.initialize():
        print("Error: Could not initialize RAG pipeline. Exiting.")
        sys.exit(1)

    print("\n========================================================")
    print("      Resume RAG Engine (Member 3) - Interactive Shell")
    print("========================================================\n")
    print(f"LLM Model       : {config.MODEL_NAME}")
    print(f"Embedding Model : {config.EMBEDDING_MODEL}")
    print(f"Re-ranker       : {config.RERANKER_MODEL if config.USE_RERANKER else 'Disabled'}")
    print(f"Ollama Server   : {config.OLLAMA_HOST}")
    print("--------------------------------------------------------")
    print("Type your question and press Enter. Type 'exit' or 'quit' to close.")
    print("========================================================\n")

    while True:
        try:
            question = input("\nQuestion: ").strip()
            if not question:
                continue
            if question.lower() in ("exit", "quit"):
                print("Exiting interactive RAG shell. Goodbye!")
                break

            response = pipeline.query(question)
            json_str = response_to_json_string(response)

            print("\nAnswer:")
            print(response["answer"])
            print("\n--------------------------------------------------------")
            print("JSON Response:")
            print(json_str)
            print("========================================================\n")

        except KeyboardInterrupt:
            print("\nExiting interactive RAG shell. Goodbye!")
            break
        except Exception as e:
            print(f"\nUnexpected error during query execution: {e}")

if __name__ == "__main__":
    main()
