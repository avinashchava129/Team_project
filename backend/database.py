#this file belongs to the database module that manages SQLite database operations for chat sessions, conversations, and uploaded file metadata.
import sqlite3
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Tuple

import config

logger = logging.getLogger(__name__)

def get_db_connection() -> sqlite3.Connection:
    """
    Ensures the parent directory exists and opens a connection to the SQLite database.
    """
    config.CHAT_HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(config.CHAT_HISTORY_PATH))
    conn.row_factory = sqlite3.Row  # Enables accessing columns by name
    return conn

def init_db() -> None:
    """
    Creates tables if they do not already exist.
    """
    logger.info("Initializing database...")
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Sessions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # 2. Conversations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources TEXT NOT NULL,          -- Serialized JSON array of string filenames
                similarity_scores TEXT NOT NULL, -- Serialized JSON array of float scores
                timestamp TEXT NOT NULL,
                FOREIGN KEY (session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
            )
        """)
        
        # 3. Uploaded Files metadata cache table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS uploaded_files (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                file_size INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL,
                chunk_count INTEGER NOT NULL
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize SQLite database: {e}")
        raise

def add_session(session_id: str, model_name: str) -> None:
    """Registers a new chat session."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().isoformat()
        cursor.execute(
            "INSERT OR IGNORE INTO sessions (session_id, model_name, created_at) VALUES (?, ?, ?)",
            (session_id, model_name, now_str)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error adding session: {e}")

def get_all_sessions() -> List[Dict[str, Any]]:
    """Returns a list of all registered sessions sorted by creation time descending."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM sessions ORDER BY created_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching sessions: {e}")
        return []

def delete_session(session_id: str) -> None:
    """Deletes a session and cascadingly deletes all its conversations."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        # Enable foreign key cascade manually in SQLite
        cursor.execute("PRAGMA foreign_keys = ON")
        cursor.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
        cursor.execute("DELETE FROM conversations WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        logger.info(f"Deleted session {session_id} and its associated chat logs.")
    except Exception as e:
        logger.error(f"Error deleting session: {e}")

def add_conversation(
    session_id: str,
    question: str,
    answer: str,
    sources: List[str],
    similarity_scores: List[float]
) -> None:
    """Saves a conversation turn to the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().isoformat()
        
        # Serialize list fields to JSON strings
        sources_json = json.dumps(sources, ensure_ascii=False)
        scores_json = json.dumps(similarity_scores)
        
        cursor.execute(
            """
            INSERT INTO conversations (session_id, question, answer, sources, similarity_scores, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (session_id, question, answer, sources_json, scores_json, now_str)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error adding conversation entry: {e}")

def get_session_conversations(session_id: str) -> List[Dict[str, Any]]:
    """Retrieves all conversation records associated with a session."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM conversations WHERE session_id = ? ORDER BY timestamp ASC", (session_id,))
        rows = cursor.fetchall()
        conn.close()
        
        conversations = []
        for row in rows:
            record = dict(row)
            # Deserialize JSON columns
            record["sources"] = json.loads(record["sources"])
            record["similarity_scores"] = json.loads(record["similarity_scores"])
            conversations.append(record)
        return conversations
    except Exception as e:
        logger.error(f"Error fetching conversation log for session {session_id}: {e}")
        return []

def add_uploaded_file(filename: str, file_size: int, chunk_count: int) -> None:
    """Registers an uploaded resume metadata inside the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        now_str = datetime.now().isoformat()
        cursor.execute(
            """
            INSERT OR REPLACE INTO uploaded_files (filename, file_size, uploaded_at, chunk_count)
            VALUES (?, ?, ?, ?)
            """,
            (filename, file_size, now_str, chunk_count)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error caching file upload metadata: {e}")

def get_uploaded_files() -> List[Dict[str, Any]]:
    """Lists all uploaded files cached in the database."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM uploaded_files ORDER BY uploaded_at DESC")
        rows = cursor.fetchall()
        conn.close()
        return [dict(row) for row in rows]
    except Exception as e:
        logger.error(f"Error fetching uploaded files: {e}")
        return []

def remove_uploaded_file(filename: str) -> None:
    """Removes a file entry from database metadata."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM uploaded_files WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error removing uploaded file entry '{filename}': {e}")

def clear_uploaded_files() -> None:
    """Clears all entries from the uploaded_files table."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM uploaded_files")
        conn.commit()
        conn.close()
    except Exception as e:
        logger.error(f"Error clearing uploaded files: {e}")
