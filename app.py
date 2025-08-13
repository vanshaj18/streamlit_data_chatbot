"""
Data Chatbot Dashboard - Main Application
A Streamlit-based web application for data analysis through natural language queries.
"""

import streamlit as st
from utils.session_manager import initialize_session
from components.file_handler import render_file_upload
from components.chat_interface import render_chat_interface


def main():
    """Main application entry point."""
    # Initialize session state
    initialize_session()
    
    # Page configuration
    st.set_page_config(
        page_title="Data Analysis Dashboard",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Sidebar with chat controls
    with st.sidebar:
        st.header("🔧 Controls")
        
        # Chat statistics
        from components.chat_interface import get_chat_stats
        stats = get_chat_stats()
        
        if stats["total_messages"] > 0:
            st.metric("Total Messages", stats["total_messages"])
            st.metric("User Questions", stats["user_messages"])
            st.metric("Agent Responses", stats["agent_messages"])
            
            # Clear chat button
            if st.button("🗑️ Clear Chat History"):
                from components.chat_interface import clear_chat
                clear_chat()
        else:
            st.info("No conversation yet. Upload data and start chatting!")
    
    # Main header
    st.title("📊 Data Chatbot Dashboard")
    st.markdown("Upload your data and ask questions in natural language!")
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("📁 Upload Data")
        render_file_upload()
    
    with col2:
        st.header("💬 Chat with Your Data")
        render_chat_interface()


if __name__ == "__main__":
    main()