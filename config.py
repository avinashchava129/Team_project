import os
from pathlib import Path

# Base project directory
BASE_DIR = Path(__file__).resolve().parent

# Data directory where PDFs are located
DATA_DIR = BASE_DIR / "data" / "resumes"

# Directory where original files might be located in the workspace (for auto-setup)
WORKSPACE_SOURCE_DIR = BASE_DIR / "Resume_File (File responses)"

# Output directory for the JSON results
OUTPUT_DIR = BASE_DIR / "output"
OUTPUT_FILE = OUTPUT_DIR / "chunks.json"

# Chunking configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
# Chunk separator hierarchy
CHUNK_SEPARATORS = ["\n\n", "\n", ".", " ", ""]

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
LOG_LEVEL = "INFO"

# --- Central Configurations ---

# Embedding settings (used by Member 2 and Member 3)
EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"

# Input path (output from Member 1)
INPUT_JSON = OUTPUT_FILE

# Vector Database Output Settings
OUTPUT_FOLDER = BASE_DIR / "vector_db"
FAISS_INDEX_NAME = "faiss_index.bin"
METADATA_NAME = "metadata.json"

# Ingestion Processing Settings (Member 2)
BATCH_SIZE = 32
NORMALIZE = True

# --- Member 3 configurations ---
# LLM API Settings (Gemini support)
USE_GEMINI = True
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-2.5-flash"

# LLM Model Name (Ollama)
MODEL_NAME = "llama3"

# Ollama Server Host
OLLAMA_HOST = "http://localhost:11434"

# Retrieval Settings
TOP_K = 5
FAISS_INDEX_PATH = OUTPUT_FOLDER / FAISS_INDEX_NAME
METADATA_PATH = OUTPUT_FOLDER / METADATA_NAME

# Re-ranking Settings
USE_RERANKER = True
RERANK_TOP_K = 3
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# LLM Generation Parameters
TEMPERATURE = 0.0  # Grounded RAG should be deterministic
MAX_NEW_TOKENS = 512
TOP_P = 0.9
TIMEOUT = 180  # Connection and inference timeout in seconds

# --- Member 4 configurations ---
APP_TITLE = "Resume RAG Assistant"
DEFAULT_MODEL = "gemini-2.5-flash"
UPLOAD_FOLDER = BASE_DIR / "uploads"
MAX_UPLOAD_SIZE = 10 * 1024 * 1024  # 10MB Limit
SUPPORTED_FORMATS = [".pdf"]
CHAT_HISTORY_DIR = BASE_DIR / "chat_history"
CHAT_HISTORY_PATH = CHAT_HISTORY_DIR / "history.db"
THEME = "dark"
API_TIMEOUT = TIMEOUT



## if  olllama is not running then we can use gemini api  and if gemini api key is not provided then we can use  ollama model  and if both are not provided then we can use  default model