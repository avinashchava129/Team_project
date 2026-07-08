import torch
import json
import logging
import requests
import shutil
import streamlit as st
from pathlib import Path

import config
from backend.session_manager import initialize_session, start_new_session
from backend.database import clear_uploaded_files

logger = logging.getLogger(__name__)

# 1. Configure Global Streamlit Page settings (must be called first)
st.set_page_config(
    page_title=config.APP_TITLE,
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Inject CSS Stylesheet
css_path = Path(__file__).resolve().parent / "assets" / "style.css"
if css_path.exists():
    try:
        with open(css_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except Exception as e:
         logger.error(f"Failed to load CSS stylesheet: {e}")

# 3. Initialize session state variables
initialize_session()

# --- SIDEBAR CONFIG PANEL ---
st.sidebar.title("⚙️ RAG settings")

@st.cache_data(ttl=300)
def get_available_models(host: str) -> list:
    options = ["gemini-2.5-flash"]
    try:
        r = requests.get(f"{host}/api/tags", timeout=1.0)
        if r.status_code == 200:
            for m in r.json().get("models", []):
                options.append(m["name"])
    except Exception:
        pass
    return options

model_options = get_available_models(config.OLLAMA_HOST)

selected_model = st.sidebar.selectbox(
    "Active LLM Backend",
    options=model_options,
    index=model_options.index(st.session_state.selected_model) if st.session_state.selected_model in model_options else 0,
    help="Switch between cloud-based Google Gemini and locally hosted Ollama models."
)

# Handle model change
if selected_model != st.session_state.selected_model:
    st.session_state.selected_model = selected_model
    logger.info(f"User switched active LLM model to: {selected_model}")
    # Force a database session record update
    from backend.database import get_db_connection
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            "UPDATE sessions SET model_name = ? WHERE session_id = ?",
            (selected_model, st.session_state.current_session_id)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Failed to update model settings in DB session: {e}")
    st.sidebar.success(f"Backend set to: {selected_model}")

st.sidebar.markdown("---")

# Render metrics in the sidebar
st.sidebar.markdown("### 📊 Status Overview")
uploaded_count = len(list(config.DATA_DIR.glob("*.pdf"))) if config.DATA_DIR.exists() else 0
st.sidebar.write(f"**Resumes:** `{uploaded_count}`")

chunks_count = 0
metadata_file = config.OUTPUT_FOLDER / config.METADATA_NAME
if metadata_file.exists():
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            chunks_count = len(json.load(f))
    except Exception:
        pass
st.sidebar.write(f"**Chunks:** `{chunks_count}`")

st.sidebar.markdown("---")

# Render Reset Action Controls
st.sidebar.markdown("### 🛠️ Actions")

if st.sidebar.button("🔄 Reset Knowledge Base", use_container_width=True, help="Deletes all PDF resumes, chunks, and FAISS vector databases"):
    try:
        # Delete source directories
        if config.DATA_DIR.exists():
            shutil.rmtree(config.DATA_DIR)
        if config.OUTPUT_DIR.exists():
            shutil.rmtree(config.OUTPUT_DIR)
        if config.OUTPUT_FOLDER.exists():
            shutil.rmtree(config.OUTPUT_FOLDER)
            
        # Reset local database metadata cache
        clear_uploaded_files()
        
        # Reset active RAG Client index context
        if "rag_client" in st.session_state:
            st.session_state.rag_client.reset_index()
            
        st.sidebar.success("Knowledge Base cleared successfully!")
        st.session_state.chat_history = []
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Reset failed: {e}")

if st.sidebar.button("🧹 Clear Chat Logs", use_container_width=True, help="Deletes all conversation log database records"):
    try:
        from backend.database import get_db_connection
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM conversations")
        cursor.execute("DELETE FROM sessions")
        conn.commit()
        conn.close()
        
        # Spawn a new session
        start_new_session(st.session_state.selected_model)
        st.sidebar.success("Audit logs cleared successfully!")
        st.rerun()
    except Exception as e:
        st.sidebar.error(f"Failed to clear database logs: {e}")

# --- STREAMLIT PAGE ROUTING SYSTEM ---
home_page = st.Page("pages/Home.py", title="Dashboard", icon="📊", default=True)
upload_page = st.Page("pages/Upload.py", title="Upload Resumes", icon="📂")
chat_page = st.Page("pages/Chat.py", title="Chat Assistant", icon="💬")
history_page = st.Page("pages/History.py", title="Conversation Logs", icon="📜")

# Assemble pages into navigation menu
pg = st.navigation([home_page, upload_page, chat_page, history_page])
pg.run()
