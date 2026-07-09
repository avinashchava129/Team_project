import json
import logging
import shutil
from pathlib import Path
from tqdm import tqdm
from typing import List, Dict, Any

import config
from utils.pdf_loader import load_pdf
from utils.text_cleaner import clean_text
from utils.chunker import generate_chunks
from utils.metadata import extract_candidate_name, create_chunk_metadata
#config"
# Configure python logging
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL, logging.INFO),
    format=config.LOG_FORMAT
)
logger = logging.getLogger(__name__)

def setup_directories() -> None:
    """
    Ensures that the input data directory and output directory exist.
    If the data directory is empty and the workspace source folder contains files,
    it automatically copies them over for immediate ingestion.
    """
    try:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    except PermissionError as e:
        logger.error(f"Permission error creating directory structure: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error initializing folders: {e}")
        raise

    # Auto-population heuristic for testing:
    # If the user has a source folder in the workspace and data directory is empty, copy them.
    if config.WORKSPACE_SOURCE_DIR.exists() and config.WORKSPACE_SOURCE_DIR.is_dir():
        try:
            existing_pdfs = list(config.DATA_DIR.glob("*.pdf"))
            if not existing_pdfs:
                logger.info(
                    f"Data directory '{config.DATA_DIR}' is empty. "
                    f"Populating from workspace folder '{config.WORKSPACE_SOURCE_DIR}'..."
                )
                for file_path in config.WORKSPACE_SOURCE_DIR.iterdir():
                    if file_path.is_file():
                        # Copy all files (we let the loader ignore non-PDFs dynamically)
                        shutil.copy2(file_path, config.DATA_DIR)
        except Exception as e:
            logger.warning(f"Could not auto-populate data folder from workspace source: {e}")

def process_resume(file_path: Path) -> List[Dict[str, Any]]:
    """
    Executes the ingestion pipeline on a single resume:
    Load -> Clean -> Chunk -> Metadata.

    Args:
        file_path: Path pointing to the PDF resume file.

    Returns:
        A list of dictionary chunks with metadata. Returns empty list if processing fails.
    """
    filename = file_path.name
    
    # 1. Load PDF text
    logger.info(f"Extracting {filename}...")
    try:
        raw_text = load_pdf(file_path)
    except Exception as e:
        logger.error(f"Error loading PDF file {filename}: {e}")
        return []

    if not raw_text or not raw_text.strip():
        logger.warning(f"Raw text is empty for {filename}. Skipping chunking.")
        return []

    # 2. Clean Text
    logger.info(f"Cleaning text for {filename}...")
    try:
        cleaned_text = clean_text(raw_text)
    except Exception as e:
        logger.error(f"Error cleaning text for {filename}: {e}")
        return []

    if not cleaned_text or not cleaned_text.strip():
        logger.warning(f"Cleaned text is empty for {filename}. Skipping chunking.")
        return []

    # 3. Create Chunks
    logger.info(f"Creating chunks for {filename}...")
    try:
        chunks = generate_chunks(
            text=cleaned_text,
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=config.CHUNK_SEPARATORS
        )
    except Exception as e:
        logger.error(f"Error splitting text for {filename}: {e}")
        return []

    if not chunks:
        logger.warning(f"No chunks generated for {filename}.")
        return []

    # 4. Generate Metadata
    candidate_name = extract_candidate_name(filename)
    total_chunks = len(chunks)
    
    resume_chunks = []
    for chunk_id, chunk_text in enumerate(chunks):
        chunk_data = create_chunk_metadata(
            candidate_name=candidate_name,
            resume_file=filename,
            chunk_id=chunk_id,
            total_chunks=total_chunks,
            text=chunk_text
        )
        resume_chunks.append(chunk_data)
        
    return resume_chunks

def main() -> None:
    """
    Main orchestrator for the Data Ingestion Pipeline.
    Loads all PDFs, extracts/cleans/splits text, creates metadata, and saves JSON.
    """
    logger.info("Loading PDFs...")
    
    try:
        setup_directories()
    except Exception as e:
        logger.critical(f"Pipeline execution aborted due to directory setup failure: {e}")
        return

    # Scan and list all files in the data directory
    try:
        all_files = sorted(config.DATA_DIR.iterdir())
    except PermissionError as e:
        logger.error(f"Permission denied accessing directory '{config.DATA_DIR}': {e}")
        return
    except Exception as e:
        logger.error(f"Failed to scan directory '{config.DATA_DIR}': {e}")
        return

    # Filter out PDFs and log warning for non-PDFs
    pdf_files = []
    for f in all_files:
        if f.is_dir():
            continue
        if f.suffix.lower() == ".pdf":
            pdf_files.append(f)
        else:
            logger.warning(f"Ignoring non-PDF file: {f.name}")

    if not pdf_files:
        logger.error(f"No PDF resumes found in directory '{config.DATA_DIR}'. Please add PDFs and try again.")
        return

    all_chunks: List[Dict[str, Any]] = []

    # Process files showing progress bar
    for file_path in tqdm(pdf_files, desc="Ingesting Resumes"):
        try:
            chunks = process_resume(file_path)
            all_chunks.extend(chunks)
        except Exception as e:
            logger.error(f"Unexpected error processing resume '{file_path.name}': {e}")

    if not all_chunks:
        logger.warning("Pipeline completed but no chunks were generated from any of the resumes.")
        return

    # 5. Save JSON
    logger.info("Saving JSON...")
    try:
        with open(config.OUTPUT_FILE, "w", encoding="utf-8") as f:
            json.dump(all_chunks, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(all_chunks)} chunks to {config.OUTPUT_FILE}.")
        logger.info("Completed successfully.")
    except PermissionError as e:
        logger.error(f"Permission denied when writing output file '{config.OUTPUT_FILE}': {e}")
    except Exception as e:
        logger.error(f"Failed to write output to file '{config.OUTPUT_FILE}': {e}")

if __name__ == "__main__":
    main()
