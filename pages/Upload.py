#upload one or multiple PDF resumes to compile them into the RAG knowledge base. The pipeline will automatically extract, clean, split, and embed the text.
import streamlit as st
from backend.uploader import process_uploads, get_uploaded_files
from backend.database import get_uploaded_files
import config

st.title("📂 Upload Resumes")
st.markdown("Upload one or multiple PDF resumes to compile them into the RAG knowledge base. The pipeline will automatically extract, clean, split, and embed the text.")

# File Uploader
uploaded_files = st.file_uploader(
    "Choose PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="Upload resumes in PDF format. Maximum file size: 10MB."
)

# Initialize processed files tracker in session state to prevent infinite loops
if "processed_filenames" not in st.session_state:
    st.session_state.processed_filenames = set()

if uploaded_files:
    current_filenames = {f.name for f in uploaded_files}
    
    # Check if the set of files in the uploader differs from what we've already compiled
    if current_filenames != st.session_state.processed_filenames:
        st.markdown("### ⚙️ Pipeline Compilation")
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        # Define progress callback to update UI
        def update_progress(percent: float, message: str):
            progress_bar.progress(percent)
            status_text.markdown(f"**Status:** {message}")
            
        success = process_uploads(uploaded_files, update_progress)
        
        if success:
            st.success("✅ Knowledge Base Updated Successfully!")
            st.session_state.kb_refresh_needed = True
            st.session_state.processed_filenames = current_filenames
            st.balloons()
            st.rerun()
        else:
            st.error("❌ Knowledge Base update failed. Please check logs.")
else:
    # If file uploader is cleared, reset the session state tracker
    st.session_state.processed_filenames = set()

st.markdown("---")

# List current resumes in store
st.markdown("### 📄 Currently Indexed Files")
db_files = get_uploaded_files()

if not db_files:
    st.info("The knowledge base is currently empty. Please upload PDF resumes to get started.")
else:
    # Print list of files in a clean grid layout
    for item in db_files:
        size_kb = round(item["file_size"] / 1024, 1)
        st.markdown(
            f"""
            <div style="background-color: rgba(255, 255, 255, 0.02); padding: 12px 18px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
                <div>
                    <span style="font-weight: 600; color: #ffffff;">📄 {item['filename']}</span>
                    <br>
                    <span style="font-size: 0.8rem; color: #888888;">Size: {size_kb} KB | Chunks: {item['chunk_count']}</span>
                </div>
                <div style="font-size: 0.8rem; color: #64748b;">Uploaded: {item['uploaded_at'][:16].replace('T', ' ')}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
