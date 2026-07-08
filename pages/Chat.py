#this file belongs to the chat page where users can ask questions about candidate resumes and receive answers with source references.
import torch
import json
import streamlit as st
from datetime import datetime

import config
from backend.session_manager import (
    initialize_session,
    save_chat_turn,
    start_new_session
)

# 1. Initialize active session states
initialize_session()

# Cache the RAG Client to maintain singleton database connections
if "rag_client" not in st.session_state:
    from backend.rag_client import RAGClient
    st.session_state.rag_client = RAGClient()

# Reset index reference if the knowledge base was rebuilt in pages/Upload.py
if st.session_state.get("kb_refresh_needed", False):
    st.session_state.rag_client.reset_index()
    st.session_state.kb_refresh_needed = False

# 2. Render Page Header
col_title, col_actions = st.columns([3, 1])
with col_title:
    st.title("💬 Resume Assistant")
    st.markdown("Ask natural language questions to search and query candidate resumes.")
with col_actions:
    # Clear conversation action
    if st.button("🗑️ Clear Conversation", use_container_width=True, help="Resets chat log and begins a new session"):
        start_new_session(st.session_state.selected_model)
        st.rerun()

st.markdown("---")

# 3. Render Historical Messages
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        # If assistant has source references, display them in an expander drawer
        if message["role"] == "assistant" and message.get("sources"):
            with st.expander("🔍 Retrieved Resume Context & Scores"):
                # Render source references as clean columns or sub-cards
                for idx, chunk in enumerate(message.get("retrieved_chunks", [])):
                    candidate = chunk.get("candidate_name", "Unknown")
                    file = chunk.get("resume_file", "Unknown")
                    score = chunk.get("rerank_score", chunk.get("score", 0.0))
                    text_excerpt = chunk.get("text", "")
                    
                    st.markdown(
                        f"""
                        <div class="source-card">
                            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                                <strong style="color: #ffffff;">📄 {file} (Candidate: {candidate})</strong>
                                <span class="score-badge">Relevance: {score:.4f}</span>
                            </div>
                            <div style="font-size: 0.9rem; color: #a1a1aa; background-color: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px; white-space: pre-wrap;">
{text_excerpt}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

# 4. Handle User Input
user_input = st.chat_input("Ask a question about candidates...")

if user_input:
    # Immediately render user question
    with st.chat_message("user"):
        st.markdown(user_input)
        
    # Render assistant typing indicator spinner
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        with st.spinner("Analyzing resumes and retrieving answers..."):
            # Call local Member 3 RAG pipeline client
            response = st.session_state.rag_client.query(
                question=user_input,
                model_name=st.session_state.selected_model
            )
            
            answer = response.get("answer", "Error retrieving answer.")
            chunks = response.get("retrieved_chunks", [])
            sources = response.get("sources", [])
            scores = response.get("similarity_scores", [])
            
            # Print response text
            response_placeholder.markdown(answer)
            
            # Display source drawer expander
            if chunks:
                with st.expander("🔍 Retrieved Resume Context & Scores"):
                    for idx, chunk in enumerate(chunks):
                        candidate = chunk.get("candidate_name", "Unknown")
                        file = chunk.get("resume_file", "Unknown")
                        score = chunk.get("rerank_score", chunk.get("score", 0.0))
                        text_excerpt = chunk.get("text", "")
                        
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 5px;">
                                    <strong style="color: #ffffff;">📄 {file} (Candidate: {candidate})</strong>
                                    <span class="score-badge">Relevance: {score:.4f}</span>
                                </div>
                                <div style="font-size: 0.9rem; color: #a1a1aa; background-color: rgba(0,0,0,0.2); padding: 8px; border-radius: 4px; white-space: pre-wrap;">
{text_excerpt}
                                </div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )
                        
        # Save exchange turn to session memory and SQLite db
        save_chat_turn(user_input, answer, sources, scores)
        st.rerun()

st.markdown("---")

# 5. Download Chat Logs Action
if st.session_state.chat_history:
    # Prepare download export string
    export_data = []
    current_q = None
    
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            current_q = msg["content"]
        elif msg["role"] == "assistant" and current_q:
            export_data.append({
                "timestamp": msg.get("timestamp", datetime.now().isoformat()),
                "question": current_q,
                "answer": msg["content"],
                "sources": msg.get("sources", []),
                "scores": msg.get("similarity_scores", [])
            })
            current_q = None
            
    json_bytes = json.dumps(export_data, indent=4, ensure_ascii=False).encode("utf-8")
    
    st.download_button(
        label="📥 Download Chat Session History",
        data=json_bytes,
        file_name=f"chat_session_{st.session_state.current_session_id[:8]}.json",
        mime="application/json",
        use_container_width=True
    )
