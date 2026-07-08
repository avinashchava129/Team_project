import json
import logging
import shutil
from pathlib import Path
from typing import List, Callable, Dict, Any

import config
import ingest
import build_vector_db
from backend.database import add_uploaded_file, get_uploaded_files

logger = logging.getLogger(__name__)

def get_file_stats(file_path: Path) -> Dict[str, Any]:
    """Returns size and stats of a file."""
    if file_path.exists():
        return {
            "size": file_path.stat().st_size,
            "modified": file_path.stat().st_mtime
        }
    return {"size": 0, "modified": 0}

def save_uploaded_file(uploaded_file, target_dir: Path) -> Path:
    """
    Saves a Streamlit UploadedFile to target directory.
    
    Args:
        uploaded_file: Streamlit UploadedFile object.
        target_dir: Directory where the file should be saved.
        
    Returns:
        Path object pointing to the saved file.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / uploaded_file.name
    
    with open(target_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    return target_path

def process_uploads(
    uploaded_files: List[Any],
    progress_callback: Callable[[float, str], None]
) -> bool:
    """
    Orchestrates the entire upload and database rebuild pipeline.
    
    1. Validates and saves PDFs to config.DATA_DIR.
    2. Runs Member 1 Ingestion programmatically (generates output/chunks.json).
    3. Runs Member 2 Embedding & Vector DB programmatically (generates index).
    4. Caches stats and updates local SQLite tables.
    
    Args:
        uploaded_files: List of Streamlit UploadedFile objects.
        progress_callback: A function taking a float (0.0 to 1.0) and a string message.
        
    Returns:
        True if the database was successfully updated, False otherwise.
    """
    if not uploaded_files:
        logger.warning("No files uploaded.")
        progress_callback(1.0, "No files uploaded.")
        return False

    # Ensure data directory exists
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # 1. Filter, validate, and save files
    new_files_saved = False
    saved_file_paths: List[Path] = []
    
    # Fetch existing files in database to check duplicates
    existing_db_files = {f["filename"] for f in get_uploaded_files()}
    
    total_files = len(uploaded_files)
    progress_callback(0.05, f"Validating and saving {total_files} file(s)...")

    for i, file_obj in enumerate(uploaded_files):
        # Validate format
        suffix = Path(file_obj.name).suffix.lower()
        if suffix not in config.SUPPORTED_FORMATS:
            logger.warning(f"Unsupported format skipped: {file_obj.name}")
            continue

        # Validate size
        if file_obj.size > config.MAX_UPLOAD_SIZE:
            logger.warning(f"File exceeds maximum size and was skipped: {file_obj.name}")
            continue

        # Save file (overwrite/prevent duplicates check)
        target_path = config.DATA_DIR / file_obj.name
        
        # Check if already exists in resumes folder (prevent duplicates)
        if target_path.exists():
            # If the file size is identical, we can skip rewriting it
            if target_path.stat().st_size == file_obj.size:
                logger.info(f"Duplicate file skipped (already exists): {file_obj.name}")
                saved_file_paths.append(target_path)
                continue
                
        # Save new file content
        logger.info(f"Saving uploaded resume: {file_obj.name}")
        save_uploaded_file(file_obj, config.DATA_DIR)
        new_files_saved = True
        saved_file_paths.append(target_path)

    # 2. Run Ingestion and FAISS Compilation programmatically
    # Even if no "new" files were saved (e.g. user just hit rebuild), we compile what is in data/resumes/
    logger.info("Running Member 1 Data Ingestion Pipeline...")
    progress_callback(0.20, "Stage 1: Extracting and cleaning text from resumes...")
    try:
        # Run Member 1
        ingest.main()
    except Exception as e:
        logger.critical(f"Ingestion pipeline failed: {e}")
        progress_callback(0.20, f"Error: Ingestion pipeline failed: {e}")
        return False

    logger.info("Running Member 2 Embedding & Vector DB compilation...")
    progress_callback(0.50, "Stage 2: Computing vector embeddings & building FAISS index...")
    try:
        # Run Member 2
        build_vector_db.main()
    except Exception as e:
        logger.critical(f"Vector Database compilation failed: {e}")
        progress_callback(0.50, f"Error: Vector store compilation failed: {e}")
        return False

    # 3. Post-process: Cache upload metrics inside SQLite
    progress_callback(0.85, "Stage 3: Updating database registry and index cache...")
    try:
        # Read compiled metadata.json to count chunk statistics for files
        metadata_file = config.OUTPUT_FOLDER / config.METADATA_NAME
        if metadata_file.exists():
            with open(metadata_file, "r", encoding="utf-8") as f:
                metadata_list = json.load(f)
                
            # Aggregate chunk counts per file
            chunk_counts: Dict[str, int] = {}
            for item in metadata_list:
                file_name = item.get("resume_file", "")
                if file_name:
                    chunk_counts[file_name] = chunk_counts.get(file_name, 0) + 1
                    
            # Register in SQLite database
            for path in config.DATA_DIR.glob("*.pdf"):
                size = path.stat().st_size
                chunks = chunk_counts.get(path.name, 0)
                add_uploaded_file(path.name, size, chunks)
                
    except Exception as e:
        logger.error(f"Error registering file stats in SQLite: {e}")

    progress_callback(1.0, "Knowledge Base Updated Successfully!")
    return True
