import json
import streamlit as st
from pathlib import Path

import config
from backend.database import get_uploaded_files

# Render page title and description
st.title("📊 System Dashboard")
st.markdown("Welcome to the **Resume QA RAG System**. This dashboard displays the live system status, active models, and knowledge base metrics.")

# 1. Gather Metrics
# Count files in resumes folder
resumes_folder = config.DATA_DIR
pdf_files = list(resumes_folder.glob("*.pdf")) if resumes_folder.exists() else []
num_resumes = len(pdf_files)

# Count chunks in index
chunks_count = 0
metadata_file = config.OUTPUT_FOLDER / config.METADATA_NAME
if metadata_file.exists():
    try:
        with open(metadata_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            chunks_count = len(meta)
    except Exception:
        pass

# Check database status
db_status = "Inactive"
db_status_color = "red"
if (config.OUTPUT_FOLDER / config.FAISS_INDEX_NAME).exists() and metadata_file.exists():
    db_status = "Active & Compiled"
    db_status_color = "green"

# Check LLM model
if config.USE_GEMINI:
    llm_info = f"Google Gemini ({config.GEMINI_MODEL})"
else:
    llm_info = f"Local Ollama ({config.MODEL_NAME})"

# 2. Render Cards
st.markdown("### 📈 Live Metrics")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Uploaded Resumes</div>
            <div class="metric-value">{num_resumes}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Semantic Chunks</div>
            <div class="metric-value">{chunks_count}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Embedding Dimension</div>
            <div class="metric-value">384</div>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown("---")

# 3. Render Status Details
col_left, col_right = st.columns(2)

with col_left:
    st.markdown("### ⚙️ System Configuration")
    
    st.markdown(f"**Vector Database:** :{db_status_color}[{db_status}]")
    st.markdown(f"**LLM Model:** `{llm_info}`")
    st.markdown(f"**Embedding Model:** `{config.EMBEDDING_MODEL}`")
    st.markdown(f"**Chunk Configuration:** Size `{config.CHUNK_SIZE}` | Overlap `{config.CHUNK_OVERLAP}`")
    st.markdown(f"**Re-ranker Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2` ({'Enabled' if config.USE_RERANKER else 'Disabled'})")

with col_right:
    st.markdown("### 📂 Indexed Candidates")
    uploaded_files_registry = get_uploaded_files()
    
    if not uploaded_files_registry:
        st.info("No candidates registered. Head over to the **Upload Resumes** page to initialize the knowledge base.")
    else:
        # Create candidate summary table
        candidates_data = []
        for index, item in enumerate(uploaded_files_registry):
            # Parse candidate name from filename
            from utils.metadata import extract_candidate_name
            candidate_name = extract_candidate_name(item["filename"])
            size_kb = round(item["file_size"] / 1024, 1)
            
            candidates_data.append({
                "No.": index + 1,
                "Candidate Name": candidate_name,
                "File Size": f"{size_kb} KB",
                "Chunks": item["chunk_count"],
                "Indexed Date": item["uploaded_at"][:10]  # Just YYYY-MM-DD
            })
            
        st.table(candidates_data)
