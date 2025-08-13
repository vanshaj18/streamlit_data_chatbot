"""
Chat interface components for natural language queries.
"""

import streamlit as st
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from typing import Any, Optional
from datetime import datetime

from utils.session_manager import (
    get_chat_history, 
    add_message, 
    has_dataframe,
    ChatMessage
)


def display_chat_history():
    """
    Display the complete chat history with proper formatting.
    
    Renders all messages in chronological order with different
    styling for user queries, agent responses, and error messages.
    """
    chat_history = get_chat_history()
    
    if not chat_history:
        st.markdown("*No messages yet. Upload data and start asking questions!*")
        return
    
    # Create a container for chat messages
    chat_container = st.container()
    
    with chat_container:
        for message in chat_history:
            _render_message(message)


def _render_message(message: ChatMessage):
    """
    Render a single chat message based on its type.
    
    Args:
        message: ChatMessage instance to render
    """
    # Format timestamp
    timestamp_str = message.timestamp.strftime("%H:%M:%S")
    
    if message.role == "user":
        # User message styling
        with st.chat_message("user"):
            st.markdown(f"**{timestamp_str}** - {message.content}")
    
    elif message.role == "agent":
        # Agent message styling
        with st.chat_message("assistant"):
            st.markdown(f"**{timestamp_str}**")
            
            if message.message_type == "text":
                st.markdown(message.content)
            
            elif message.message_type == "plot":
                st.markdown(message.content)
                if message.chart_data is not None:
                    _render_chart(message.chart_data)
            
            elif message.message_type == "error":
                st.error(f"Error: {message.content}")


def _render_chart(chart_data: Any):
    """
    Render chart data based on its type.
    
    Args:
        chart_data: Chart data (matplotlib figure or plotly figure)
    """
    try:
        # Handle matplotlib figures
        if hasattr(chart_data, 'savefig'):
            st.pyplot(chart_data)
        
        # Handle plotly figures
        elif hasattr(chart_data, 'show'):
            st.plotly_chart(chart_data, use_container_width=True)
        
        # Handle other chart types or raw data
        else:
            st.write(chart_data)
            
    except Exception as e:
        st.error(f"Failed to render chart: {str(e)}")


def handle_user_input():
    """
    Handle user input from the chat interface.
    
    Processes the user's query, validates that data is loaded,
    and adds the message to chat history. Returns the query
    for further processing by the PandasAI agent.
    
    Returns:
        Optional[str]: The user's query if valid, None otherwise
    """
    # Check if data is loaded
    if not has_dataframe():
        st.warning("Please upload a dataset before asking questions.")
        return None
    
    # Create input field with unique key to prevent conflicts
    user_query = st.chat_input("Ask a question about your data...")
    
    if user_query:
        # Add user message to chat history
        add_message(
            role="user",
            content=user_query,
            message_type="text"
        )
        
        # Return the query for processing
        return user_query
    
    return None


def display_response(content: str, message_type: str = "text", 
                    chart_data: Optional[Any] = None):
    """
    Display an agent response and add it to chat history.
    
    Args:
        content: The response content/message
        message_type: Type of response ("text", "plot", "error")
        chart_data: Optional chart data for visualizations
    """
    # Add to chat history
    add_message(
        role="agent",
        content=content,
        message_type=message_type,
        chart_data=chart_data
    )
    
    # Force a rerun to display the new message
    st.rerun()


def display_error_message(error_msg: str):
    """
    Display an error message in the chat interface.
    
    Args:
        error_msg: The error message to display
    """
    display_response(error_msg, message_type="error")


def display_text_response(response: str):
    """
    Display a text response from the agent.
    
    Args:
        response: The text response to display
    """
    display_response(response, message_type="text")


def display_chart_response(response: str, chart_data: Any):
    """
    Display a chart response from the agent.
    
    Args:
        response: Description or context for the chart
        chart_data: The chart data to display
    """
    display_response(response, message_type="plot", chart_data=chart_data)


def render_chat_interface():
    """
    Render the complete chat interface component.
    
    This is the main function that orchestrates the chat interface,
    displaying chat history and handling user input.
    """
    # Display chat history
    display_chat_history()
    
    # Handle user input
    user_query = handle_user_input()
    
    # If there's a new query, it will be processed by the main app
    # The actual query processing will be handled in task 5 (PandasAI integration)
    if user_query:
        # For now, show a placeholder response since PandasAI isn't integrated yet
        st.info("Query received! PandasAI integration will be implemented in task 5.")
        
        # Add a placeholder response to demonstrate the interface
        add_message(
            role="agent",
            content="I received your query, but PandasAI integration is not yet implemented. This will be completed in task 5.",
            message_type="text"
        )
        st.rerun()


def clear_chat():
    """
    Clear the chat history.
    
    Provides a way to reset the conversation while keeping the data loaded.
    """
    from utils.session_manager import clear_chat_history
    clear_chat_history()
    st.rerun()


def get_chat_stats():
    """
    Get statistics about the current chat session.
    
    Returns:
        dict: Dictionary containing chat statistics
    """
    chat_history = get_chat_history()
    
    user_messages = [msg for msg in chat_history if msg.role == "user"]
    agent_messages = [msg for msg in chat_history if msg.role == "agent"]
    error_messages = [msg for msg in chat_history if msg.message_type == "error"]
    plot_messages = [msg for msg in chat_history if msg.message_type == "plot"]
    
    return {
        "total_messages": len(chat_history),
        "user_messages": len(user_messages),
        "agent_messages": len(agent_messages),
        "error_messages": len(error_messages),
        "plot_messages": len(plot_messages),
        "session_start": chat_history[0].timestamp if chat_history else None,
        "last_message": chat_history[-1].timestamp if chat_history else None
    }