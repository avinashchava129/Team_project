import logging
import fitz  # PyMuPDF
from pathlib import Path
from typing import Dict
"pdf loading"
logger = logging.getLogger(__name__)

def load_pdf(pdf_path: Path) -> str:
    """
    Extracts text from a single PDF file using PyMuPDF (fitz).

    Args:
        pdf_path: Path object pointing to the PDF file.

    Returns:
        The extracted raw text string. Returns an empty string if extraction fails,
        the file is empty, or cannot be parsed.
    """
    if not pdf_path.is_file():
        logger.warning(f"File not found or is not a file: {pdf_path}")
        return ""

    if pdf_path.suffix.lower() != ".pdf":
        logger.warning(f"Skipping non-PDF file: {pdf_path.name}")
        return ""

    try:
        # Open  PDF document
        doc = fitz.open(pdf_path)
    except Exception as e:
        logger.error(f"Error opening or reading corrupted PDF {pdf_path.name}: {e}")
        return ""

    text_pages = []
    try:
        # Loop through pages and extract text
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text:
                text_pages.append(page_text)
    except Exception as e:
        logger.error(f"Error extracting text from PDF {pdf_path.name} at page {page_num}: {e}")
    finally:
        doc.close()

    full_text = "\n".join(text_pages)
    if not full_text.strip():
        logger.warning(f"No text extracted from PDF {pdf_path.name}. Attempting OCR fallback using easyocr...")
        try:
            import easyocr
            import numpy as np
            
            # Re-open doc for page rendering
            doc = fitz.open(pdf_path)
            ocr_pages = []
            
            # Initialize reader (CPU only by default to avoid CUDA initialization conflicts)
            reader = easyocr.Reader(['en'], gpu=False)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
                result = reader.readtext(img_data)
                page_text = "\n".join([item[1] for item in result])
                if page_text:
                    ocr_pages.append(page_text)
            
            doc.close()
            full_text = "\n".join(ocr_pages)
            if full_text.strip():
                logger.info(f"Successfully extracted {len(full_text)} characters using OCR fallback for {pdf_path.name}.")
            else:
                logger.warning(f"OCR fallback yielded no text for {pdf_path.name}.")
        except Exception as ocr_err:
            logger.error(f"OCR fallback failed for {pdf_path.name}: {ocr_err}")

    return full_text

def load_all_pdfs(directory: Path) -> Dict[str, str]:
    """
    Scan the specified directory for PDF files, extract text from each,
    and return a mapping from filename to extracted text.
    Non-PDF files are ignored, and corrupted/empty PDFs are handled gracefully.

    Args:
        directory: Path object to the directory containing PDFs.

    Returns:
        A dictionary mapping PDF filenames to their raw extracted text.
    """
    if not directory.is_dir():
        logger.error(f"Provided path is not a directory: {directory}")
        return {}

    pdf_texts = {}
    
    # Sort files to ensure deterministic ingestion order
    all_files = sorted(directory.iterdir())
    
    for file_path in all_files:
        if file_path.is_dir():
            continue

        if file_path.suffix.lower() != ".pdf":
            logger.info(f"Skipping non-PDF file: {file_path.name}")
            continue

        logger.info(f"Extracting {file_path.name}...")
        extracted_text = load_pdf(file_path)
        pdf_texts[file_path.name] = extracted_text

    return pdf_texts
