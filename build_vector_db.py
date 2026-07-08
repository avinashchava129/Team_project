import logging
from pathlib import Path

import config
from utils.data_loader import load_chunks_from_json
from utils.embedding_model import EmbeddingEngine
from utils.vector_store import save_vector_store

# Configure python logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

def main() -> None:
    """
    Main orchestrator for Member 2:
    Loads chunks -> Computes Embeddings -> Normalizes -> Saves FAISS index and metadata.
    """
    logger.info("Initializing Embeddings & Vector Database Pipeline...")

    # 1. Load JSON chunks
    logger.info("Loading chunks...")
    try:
        chunks = load_chunks_from_json(config.INPUT_JSON)
    except Exception as e:
        logger.critical(f"Unexpected crash during chunk loading: {e}")
        return

    if not chunks:
        logger.error("No valid chunks were loaded. Pipeline cannot continue.")
        return

    # Calculate statistics
    unique_documents = {chunk["resume_file"] for chunk in chunks}
    num_docs = len(unique_documents)
    num_chunks = len(chunks)

    # 2. Load Embedding Model
    logger.info("Loading embedding model...")
    try:
        engine = EmbeddingEngine(model_name=config.EMBEDDING_MODEL)
    except Exception as e:
        logger.critical(f"Embedding model loading failure: {e}")
        return

    # Extract text contents
    texts = [chunk["text"] for chunk in chunks]

    # 3. Generate & 4. Normalize Embeddings
    logger.info("Generating embeddings...")
    try:
        embeddings = engine.generate_embeddings(
            texts=texts,
            batch_size=config.BATCH_SIZE,
            normalize=config.NORMALIZE
        )
    except Exception as e:
        logger.critical(f"Embedding generation failure: {e}")
        return

    embedding_dim = embeddings.shape[1]

    # 5-8. Create FAISS, Insert, and Save to disk
    logger.info("Saving vector database...")
    try:
        save_vector_store(
            embeddings=embeddings,
            chunks=chunks,
            output_folder=config.OUTPUT_FOLDER,
            index_name=config.FAISS_INDEX_NAME,
            metadata_name=config.METADATA_NAME,
            normalize=config.NORMALIZE
        )
    except Exception as e:
        logger.critical(f"FAISS save failure: {e}")
        return

    # 9. Print Summary in requested format
    print("\n---")
    print(f"Documents Processed : {num_docs}")
    print(f"Chunks              : {num_chunks}")
    print(f"Embedding Dimension : {embedding_dim}")
    print("Vector Database Saved")
    print("---\n")
    
    logger.info("Completed successfully.")

if __name__ == "__main__":
    main()
