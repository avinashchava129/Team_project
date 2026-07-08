#this file belongs to the history page where users can search, inspect, delete, or download historical resume QA exchanges from the database.
import json
import sqlite3
import streamlit as st

import config
from backend.database import get_db_connection, get_all_sessions
from backend.session_manager import delete_session, load_session

st.title("📜 Conversation Logs")
st.markdown("Search, inspect, delete, or download historical resume QA exchanges from the database.")

# 1. Global Search Interface
st.markdown("### 🔍 Global Search Logs")
search_query = st.text_input("Enter search keywords...", help="Filters conversations across all sessions by matching question or answer text.")

if search_query:
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.*, s.model_name 
            FROM conversations c 
            JOIN sessions s ON c.session_id = s.session_id 
            WHERE c.question LIKE ? OR c.answer LIKE ?
            ORDER BY c.timestamp DESC
            """,
            (f"%{search_query}%", f"%{search_query}%")
        )
        results = cursor.fetchall()
        conn.close()
        
        if not results:
            st.warning(f"No logged matches found for keywords: '{search_query}'")
        else:
            st.success(f"Found {len(results)} matching conversation(s):")
            
            for row in results:
                record = dict(row)
                st.markdown(
                    f"""
                    <div style="background-color: rgba(255, 255, 255, 0.02); padding: 15px; border-radius: 8px; border: 1px solid rgba(255, 255, 255, 0.05); margin-bottom: 12px;">
                        <div style="font-size: 0.8rem; color: #64748b; margin-bottom: 5px; display: flex; justify-content: space-between;">
                            <span>Session ID: {record['session_id'][:8]}... | Model: {record['model_name']}</span>
                            <span>{record['timestamp'].replace('T', ' ')[:19]}</span>
                        </div>
                        <p style="color: #3b82f6; font-weight: 600; margin: 0 0 5px 0;">Q: {record['question']}</p>
                        <p style="color: #e2e8f0; margin: 0;">A: {record['answer']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(f"Failed to query SQLite search: {e}")

st.markdown("---")

# 2. Browse Session Logs
st.markdown("### 📁 Browse Chat Sessions")
sessions_list = get_all_sessions()

if not sessions_list:
    st.info("No chat logs are saved in the database.")
else:
    # Prepare session list descriptions
    session_options = {}
    for idx, sess in enumerate(sessions_list):
        date_str = sess["created_at"][:16].replace("T", " ")
        label = f"Session {sess['session_id'][:8]}... ({sess['model_name']} | {date_str})"
        session_options[sess["session_id"]] = label
        
    selected_sess_id = st.selectbox(
        "Select a chat session to review",
        options=list(session_options.keys()),
        format_func=lambda x: session_options[x]
    )
    
    if selected_sess_id:
        # Load and display conversation exchanges
        from backend.database import get_session_conversations
        convs = get_session_conversations(selected_sess_id)
        
        if not convs:
            st.info("No conversation exchanges recorded in this session.")
        else:
            col_actions_left, col_actions_right = st.columns(2)
            
            with col_actions_left:
                # Load session into active chatbot screen page
                if st.button("💬 Restore Session to Chat Page", use_container_width=True):
                    success = load_session(selected_sess_id)
                    if success:
                        st.success("Session loaded! Navigate to the Chat page to resume.")
                        st.balloons()
                        
            with col_actions_right:
                # Delete session log
                if st.button("🗑️ Delete Session Log", use_container_width=True, type="secondary"):
                    delete_session(selected_sess_id)
                    st.success("Deleted chat session log!")
                    st.rerun()

            st.markdown("#### Conversation Logs:")
            for index, conv in enumerate(convs):
                st.markdown(f"**Turn {index + 1}**")
                
                # Question bubble
                st.markdown(
                    f"""
                    <div style="background-color: rgba(30, 41, 59, 0.4); padding: 12px 18px; border-radius: 8px; border-left: 4px solid #3b82f6; margin-bottom: 8px;">
                        <span style="font-size: 0.8rem; color: #3b82f6; font-weight: 600; text-transform: uppercase;">User</span>
                        <div style="margin-top: 5px; color: #e2e8f0;">{conv['question']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Answer bubble
                st.markdown(
                    f"""
                    <div style="background-color: rgba(15, 23, 42, 0.4); padding: 12px 18px; border-radius: 8px; border-left: 4px solid #10b981; margin-bottom: 15px;">
                        <span style="font-size: 0.8rem; color: #10b981; font-weight: 600; text-transform: uppercase;">Assistant</span>
                        <div style="margin-top: 5px; color: #e2e8f0; white-space: pre-wrap;">{conv['answer']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
            # Session JSON Export button
            json_bytes = json.dumps(convs, indent=4, ensure_ascii=False).encode("utf-8")
            st.download_button(
                label="📥 Export Session Log to JSON",
                data=json_bytes,
                file_name=f"exported_session_{selected_sess_id[:8]}.json",
                mime="application/json",
                use_container_width=True
            )
