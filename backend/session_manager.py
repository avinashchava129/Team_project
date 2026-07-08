import uuid
import logging
import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional

import config
from backend.database import (
    init_db,
    add_session,
    get_all_sessions,
    delete_session as db_delete_session,
    add_conversation,
    get_session_conversations
)

logger = logging.getLogger(__name__)

def initialize_session() -> None:
    """
    Initializes required session state variables inside Streamlit.
    Connects to and initializes the database on first load.
    """
    # 1. Initialize SQLite Database
    if "db_initialized" not in st.session_state:
        try:
            init_db()
            st.session_state.db_initialized = True
        except Exception as e:
            logger.error(f"Error initializing DB during session start: {e}")
            st.session_state.db_initialized = False

    # 2. Setup Session State variables
    if "current_session_id" not in st.session_state:
        st.session_state.current_session_id = None
        
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []  # List of dicts: {"role": "user"/"assistant", "content": "...", "metadata": {}}
        
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = config.DEFAULT_MODEL

    if "kb_refresh_needed" not in st.session_state:
        st.session_state.kb_refresh_needed = False

    # 3. Create default session if none exists
    if st.session_state.current_session_id is None:
        start_new_session(st.session_state.selected_model)

def start_new_session(model_name: str) -> str:
    """
    Generates a unique session ID, registers it in the database,
    and resets the in-memory chat history.

    Args:
        model_name: Name of the LLM model selected for the session.

    Returns:
        The generated session ID string.
    """
    session_id = str(uuid.uuid4())
    logger.info(f"Starting new chat session: {session_id} using {model_name}")
    
    # Save to SQLite database
    add_session(session_id, model_name)
    
    # Set active session states
    st.session_state.current_session_id = session_id
    st.session_state.chat_history = []
    st.session_state.selected_model = model_name
    
    return session_id

def load_session(session_id: str) -> bool:
    """
    Loads conversation history for a specific session ID from the database into memory.

    Args:
        session_id: Target session ID to load.

    Returns:
        True if loaded successfully, False otherwise.
    """
    logger.info(f"Loading session {session_id} from database...")
    try:
        conversations = get_session_conversations(session_id)
        
        # Reconstruct chat_history list format
        history = []
        for conv in conversations:
            # User question
            history.append({
                "role": "user",
                "content": conv["question"],
                "timestamp": conv["timestamp"]
            })
            # Assistant answer
            history.append({
                "role": "assistant",
                "content": conv["answer"],
                "timestamp": conv["timestamp"],
                "sources": conv["sources"],
                "similarity_scores": conv["similarity_scores"]
            })
            
        st.session_state.current_session_id = session_id
        st.session_state.chat_history = history
        
        # Load the session's model name
        sessions = get_all_sessions()
        for sess in sessions:
            if sess["session_id"] == session_id:
                st.session_state.selected_model = sess["model_name"]
                break
                
        return True
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        return False

def delete_session(session_id: str) -> None:
    """
    Deletes a session from the DB and resets the active session if it matches the deleted one.

    Args:
        session_id: Target session ID to delete.
    """
    db_delete_session(session_id)
    
    # If the user deleted the active session, start a new one
    if st.session_state.current_session_id == session_id:
        st.session_state.current_session_id = None
        st.session_state.chat_history = []
        start_new_session(st.session_state.selected_model)

def save_chat_turn(
    question: str,
    answer: str,
    sources: List[str],
    similarity_scores: List[float]
) -> None:
    """
    Appends the question-answer turn to the active in-memory chat history
    and persists it in SQLite.
    """
    session_id = st.session_state.current_session_id
    if not session_id:
        logger.warning("No active session ID found. Starting a new session to save chat turn.")
        session_id = start_new_session(st.session_state.selected_model)
        
    now_str = datetime.now().isoformat()
    
    # 1. Update in-memory chat history
    st.session_state.chat_history.append({
        "role": "user",
        "content": question,
        "timestamp": now_str
    })
    st.session_state.chat_history.append({
        "role": "assistant",
        "content": answer,
        "timestamp": now_str,
        "sources": sources,
        "similarity_scores": similarity_scores
    })
    
    # 2. Persist to SQLite
    add_conversation(session_id, question, answer, sources, similarity_scores)
