"""
Session state management utilities.
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime


@dataclass
class ChatMessage:
    """Data class for chat messages."""
    role: str  # "user" or "agent"
    content: str
    timestamp: datetime
    message_type: str  # "text", "plot", "error"
    chart_data: Optional[Any] = None


@dataclass
class FileInfo:
    """Data class for file information."""
    filename: str
    file_size: int
    upload_timestamp: datetime
    columns: List[str]
    row_count: int
    file_type: str


@dataclass
class SessionState:
    """Data class for session state."""
    dataframe: Optional[pd.DataFrame] = None
    chat_history: List[ChatMessage] = field(default_factory=list)
    agent: Optional[Any] = None
    file_info: Optional[FileInfo] = None
    is_initialized: bool = False


def initialize_session():
    """
    Initialize session state variables.
    
    Creates a new SessionState instance if one doesn't exist.
    This ensures the application starts with a clean state.
    """
    if 'session_data' not in st.session_state:
        st.session_state.session_data = SessionState()
        st.session_state.session_data.is_initialized = True


def get_session_data() -> SessionState:
    """
    Get current session data.
    
    Returns:
        SessionState: Current session state instance
    """
    if 'session_data' not in st.session_state:
        initialize_session()
    return st.session_state.session_data


def update_dataframe(dataframe: pd.DataFrame, file_info: Dict[str, Any]) -> None:
    """
    Store uploaded data in session.
    
    Args:
        dataframe: The pandas DataFrame to store
        file_info: Dictionary containing file metadata
    """
    session_data = get_session_data()
    
    # Store the DataFrame
    session_data.dataframe = dataframe
    
    # Create FileInfo object from dictionary
    session_data.file_info = FileInfo(
        filename=file_info.get('filename', 'unknown'),
        file_size=file_info.get('file_size', 0),
        upload_timestamp=file_info.get('upload_timestamp', datetime.now()),
        columns=list(dataframe.columns) if dataframe is not None else [],
        row_count=len(dataframe) if dataframe is not None else 0,
        file_type=file_info.get('file_type', 'unknown')
    )
    
    # Reset agent when new data is uploaded
    session_data.agent = None


def add_message(role: str, content: str, message_type: str = "text", 
                chart_data: Optional[Any] = None) -> None:
    """
    Append a message to chat history.
    
    Args:
        role: Either "user" or "agent"
        content: The message content
        message_type: Type of message ("text", "plot", "error")
        chart_data: Optional chart data for visualizations
    """
    session_data = get_session_data()
    
    message = ChatMessage(
        role=role,
        content=content,
        timestamp=datetime.now(),
        message_type=message_type,
        chart_data=chart_data
    )
    
    session_data.chat_history.append(message)


def get_chat_history() -> List[ChatMessage]:
    """
    Get the current chat history.
    
    Returns:
        List[ChatMessage]: List of chat messages
    """
    session_data = get_session_data()
    return session_data.chat_history


def has_dataframe() -> bool:
    """
    Check if a DataFrame is loaded in the session.
    
    Returns:
        bool: True if DataFrame exists, False otherwise
    """
    session_data = get_session_data()
    return session_data.dataframe is not None


def get_dataframe() -> Optional[pd.DataFrame]:
    """
    Get the current DataFrame from session.
    
    Returns:
        Optional[pd.DataFrame]: The current DataFrame or None
    """
    session_data = get_session_data()
    return session_data.dataframe


def get_file_info() -> Optional[FileInfo]:
    """
    Get the current file information.
    
    Returns:
        Optional[FileInfo]: File information or None
    """
    session_data = get_session_data()
    return session_data.file_info


def set_agent(agent: Any) -> None:
    """
    Store the PandasAI agent in session.
    
    Args:
        agent: The PandasAI agent instance
    """
    session_data = get_session_data()
    session_data.agent = agent


def get_agent() -> Optional[Any]:
    """
    Get the current PandasAI agent.
    
    Returns:
        Optional[Any]: The agent instance or None
    """
    session_data = get_session_data()
    return session_data.agent


def clear_session() -> None:
    """
    Reset application state.
    
    Clears all session data including DataFrame, chat history,
    agent, and file information.
    """
    if 'session_data' in st.session_state:
        del st.session_state.session_data
    initialize_session()


def clear_chat_history() -> None:
    """
    Clear only the chat history while preserving other session data.
    """
    session_data = get_session_data()
    session_data.chat_history.clear()


def get_session_summary() -> Dict[str, Any]:
    """
    Get a summary of the current session state.
    
    Returns:
        Dict[str, Any]: Summary of session state
    """
    session_data = get_session_data()
    
    return {
        'has_dataframe': session_data.dataframe is not None,
        'dataframe_shape': session_data.dataframe.shape if session_data.dataframe is not None else None,
        'chat_message_count': len(session_data.chat_history),
        'has_agent': session_data.agent is not None,
        'file_info': {
            'filename': session_data.file_info.filename if session_data.file_info else None,
            'file_size': session_data.file_info.file_size if session_data.file_info else None,
            'row_count': session_data.file_info.row_count if session_data.file_info else None,
            'column_count': len(session_data.file_info.columns) if session_data.file_info else None
        } if session_data.file_info else None,
        'is_initialized': session_data.is_initialized
    }