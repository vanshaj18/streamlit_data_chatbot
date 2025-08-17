"""
Session state management utilities with memory optimization.
"""

import streamlit as st
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
import sys
import gc
import logging

logger = logging.getLogger(__name__)

# Configuration for memory management
MAX_CHAT_HISTORY = 100  # Maximum number of chat messages to keep
MEMORY_CLEANUP_THRESHOLD = 50  # Clean up memory every N operations


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
    Append a message to chat history with memory management.
    
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
    
    # Manage chat history size to prevent memory issues
    if len(session_data.chat_history) > MAX_CHAT_HISTORY:
        # Remove oldest messages, keeping the most recent ones
        messages_to_remove = len(session_data.chat_history) - MAX_CHAT_HISTORY
        removed_messages = session_data.chat_history[:messages_to_remove]
        session_data.chat_history = session_data.chat_history[messages_to_remove:]
        
        # Clean up chart data from removed messages to free memory
        for msg in removed_messages:
            if msg.chart_data is not None:
                del msg.chart_data
        
        logger.info(f"Removed {messages_to_remove} old messages to manage memory")
    
    # Periodic memory cleanup
    if len(session_data.chat_history) % MEMORY_CLEANUP_THRESHOLD == 0:
        _cleanup_memory()


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
    Get a summary of the current session state with memory usage info.
    
    Returns:
        Dict[str, Any]: Summary of session state
    """
    session_data = get_session_data()
    
    # Calculate memory usage
    memory_info = _get_memory_usage()
    
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
        'is_initialized': session_data.is_initialized,
        'memory_usage': memory_info
    }


def _cleanup_memory() -> None:
    """
    Perform memory cleanup operations.
    """
    try:
        # Force garbage collection
        gc.collect()
        
        # Clear visualization cache if it exists
        try:
            from components.visualization import clear_chart_cache
            clear_chart_cache()
        except ImportError:
            pass
        
        logger.debug("Memory cleanup completed")
        
    except Exception as e:
        logger.warning(f"Memory cleanup failed: {str(e)}")


def _get_memory_usage() -> Dict[str, Any]:
    """
    Get current memory usage statistics.
    
    Returns:
        Dict with memory usage information
    """
    try:
        session_data = get_session_data()
        
        # Calculate approximate memory usage
        dataframe_memory = 0
        if session_data.dataframe is not None:
            dataframe_memory = session_data.dataframe.memory_usage(deep=True).sum()
        
        chat_memory = sys.getsizeof(session_data.chat_history)
        for msg in session_data.chat_history:
            chat_memory += sys.getsizeof(msg.content)
            if msg.chart_data is not None:
                chat_memory += sys.getsizeof(msg.chart_data)
        
        return {
            'dataframe_memory_mb': round(dataframe_memory / (1024 * 1024), 2),
            'chat_memory_mb': round(chat_memory / (1024 * 1024), 2),
            'total_messages': len(session_data.chat_history),
            'max_messages': MAX_CHAT_HISTORY
        }
        
    except Exception as e:
        logger.warning(f"Failed to calculate memory usage: {str(e)}")
        return {
            'dataframe_memory_mb': 0,
            'chat_memory_mb': 0,
            'total_messages': 0,
            'max_messages': MAX_CHAT_HISTORY
        }


def optimize_session_memory() -> Dict[str, Any]:
    """
    Optimize session memory usage and return statistics.
    
    Returns:
        Dict with optimization results
    """
    try:
        session_data = get_session_data()
        
        # Get initial memory usage
        initial_memory = _get_memory_usage()
        
        # Optimize DataFrame if present
        if session_data.dataframe is not None:
            original_memory = session_data.dataframe.memory_usage(deep=True).sum()
            
            # Optimize data types (similar to file_handler optimization)
            session_data.dataframe = _optimize_dataframe_memory(session_data.dataframe)
            
            optimized_memory = session_data.dataframe.memory_usage(deep=True).sum()
            memory_saved = original_memory - optimized_memory
            
            logger.info(f"DataFrame memory optimized: saved {memory_saved / (1024*1024):.2f} MB")
        
        # Clean up old chart data in messages
        chart_data_cleaned = 0
        for msg in session_data.chat_history[:-20]:  # Keep recent 20 messages with charts
            if msg.chart_data is not None:
                del msg.chart_data
                msg.chart_data = None
                chart_data_cleaned += 1
        
        # Force cleanup
        _cleanup_memory()
        
        # Get final memory usage
        final_memory = _get_memory_usage()
        
        return {
            'initial_memory': initial_memory,
            'final_memory': final_memory,
            'chart_data_cleaned': chart_data_cleaned,
            'optimization_successful': True
        }
        
    except Exception as e:
        logger.error(f"Session memory optimization failed: {str(e)}")
        return {
            'optimization_successful': False,
            'error': str(e)
        }


def _optimize_dataframe_memory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize DataFrame memory usage by downcasting numeric types.
    
    Args:
        df: DataFrame to optimize
        
    Returns:
        Optimized DataFrame
    """
    try:
        optimized_df = df.copy()
        
        # Downcast numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='integer')
        
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Convert low-cardinality object columns to category
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
        
        return optimized_df
        
    except Exception as e:
        logger.warning(f"DataFrame memory optimization failed: {str(e)}")
        return df