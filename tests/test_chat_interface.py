"""
Unit tests for chat interface components.
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock

# Import the components to test
from components.chat_interface import (
    display_chat_history,
    handle_user_input,
    display_response,
    display_error_message,
    display_text_response,
    display_chart_response,
    clear_chat,
    get_chat_stats,
    _render_message,
    _render_chart
)
from utils.session_manager import ChatMessage, initialize_session, clear_session


class TestChatInterface:
    """Test suite for chat interface components."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Mock streamlit to avoid actual UI rendering during tests
        self.st_mock = Mock()
        
        # Clear session state before each test
        clear_session()
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface.get_chat_history')
    def test_display_chat_history_empty(self, mock_get_history, mock_st):
        """Test displaying empty chat history."""
        mock_get_history.return_value = []
        mock_st.container.return_value.__enter__ = Mock()
        mock_st.container.return_value.__exit__ = Mock()
        
        display_chat_history()
        
        mock_st.markdown.assert_called_with("*No messages yet. Upload data and start asking questions!*")
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface.get_chat_history')
    @patch('components.chat_interface._render_message')
    def test_display_chat_history_with_messages(self, mock_render, mock_get_history, mock_st):
        """Test displaying chat history with messages."""
        # Create test messages
        test_messages = [
            ChatMessage(
                role="user",
                content="Test question",
                timestamp=datetime.now(),
                message_type="text"
            ),
            ChatMessage(
                role="agent",
                content="Test response",
                timestamp=datetime.now(),
                message_type="text"
            )
        ]
        
        mock_get_history.return_value = test_messages
        mock_st.container.return_value.__enter__ = Mock()
        mock_st.container.return_value.__exit__ = Mock()
        
        display_chat_history()
        
        # Verify _render_message was called for each message
        assert mock_render.call_count == 2
        mock_render.assert_any_call(test_messages[0])
        mock_render.assert_any_call(test_messages[1])
    
    @patch('components.chat_interface.st')
    def test_render_user_message(self, mock_st):
        """Test rendering a user message."""
        message = ChatMessage(
            role="user",
            content="What is the average age?",
            timestamp=datetime(2024, 1, 1, 12, 0, 0),
            message_type="text"
        )
        
        mock_chat_message = Mock()
        mock_st.chat_message.return_value = mock_chat_message
        mock_chat_message.__enter__ = Mock()
        mock_chat_message.__exit__ = Mock()
        
        _render_message(message)
        
        mock_st.chat_message.assert_called_with("user")
        mock_st.markdown.assert_called_with("**12:00:00** - What is the average age?")
    
    @patch('components.chat_interface.st')
    def test_render_agent_text_message(self, mock_st):
        """Test rendering an agent text message."""
        message = ChatMessage(
            role="agent",
            content="The average age is 35.2 years",
            timestamp=datetime(2024, 1, 1, 12, 0, 5),
            message_type="text"
        )
        
        mock_chat_message = Mock()
        mock_st.chat_message.return_value = mock_chat_message
        mock_chat_message.__enter__ = Mock()
        mock_chat_message.__exit__ = Mock()
        
        _render_message(message)
        
        mock_st.chat_message.assert_called_with("assistant")
        # Should call markdown twice - once for timestamp, once for content
        assert mock_st.markdown.call_count == 2
        mock_st.markdown.assert_any_call("**12:00:05**")
        mock_st.markdown.assert_any_call("The average age is 35.2 years")
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface._render_chart')
    def test_render_agent_plot_message(self, mock_render_chart, mock_st):
        """Test rendering an agent plot message."""
        mock_chart_data = Mock()
        message = ChatMessage(
            role="agent",
            content="Here's a chart showing the data distribution",
            timestamp=datetime(2024, 1, 1, 12, 0, 10),
            message_type="plot",
            chart_data=mock_chart_data
        )
        
        mock_chat_message = Mock()
        mock_st.chat_message.return_value = mock_chat_message
        mock_chat_message.__enter__ = Mock()
        mock_chat_message.__exit__ = Mock()
        
        _render_message(message)
        
        mock_st.chat_message.assert_called_with("assistant")
        mock_render_chart.assert_called_with(mock_chart_data)
    
    @patch('components.chat_interface.st')
    def test_render_agent_error_message(self, mock_st):
        """Test rendering an agent error message."""
        message = ChatMessage(
            role="agent",
            content="Unable to process query",
            timestamp=datetime(2024, 1, 1, 12, 0, 15),
            message_type="error"
        )
        
        mock_chat_message = Mock()
        mock_st.chat_message.return_value = mock_chat_message
        mock_chat_message.__enter__ = Mock()
        mock_chat_message.__exit__ = Mock()
        
        _render_message(message)
        
        mock_st.chat_message.assert_called_with("assistant")
        mock_st.error.assert_called_with("Error: Unable to process query")
    
    @patch('components.chat_interface.st')
    def test_render_matplotlib_chart(self, mock_st):
        """Test rendering a matplotlib chart."""
        # Create a mock matplotlib figure
        mock_fig = Mock()
        mock_fig.savefig = Mock()  # This makes it identifiable as matplotlib
        
        _render_chart(mock_fig)
        
        mock_st.pyplot.assert_called_with(mock_fig)
    
    @patch('components.chat_interface.st')
    def test_render_plotly_chart(self, mock_st):
        """Test rendering a plotly chart."""
        # Create a mock plotly figure
        mock_fig = Mock()
        mock_fig.show = Mock()  # This makes it identifiable as plotly
        # Remove savefig to ensure it's not treated as matplotlib
        if hasattr(mock_fig, 'savefig'):
            delattr(mock_fig, 'savefig')
        
        _render_chart(mock_fig)
        
        mock_st.plotly_chart.assert_called_with(mock_fig, use_container_width=True)
    
    @patch('components.chat_interface.st')
    def test_render_chart_fallback(self, mock_st):
        """Test rendering chart with fallback for unknown types."""
        mock_data = {"type": "unknown", "data": [1, 2, 3]}
        
        _render_chart(mock_data)
        
        mock_st.write.assert_called_with(mock_data)
    
    @patch('components.chat_interface.st')
    def test_render_chart_error_handling(self, mock_st):
        """Test chart rendering error handling."""
        mock_fig = Mock()
        mock_fig.savefig = Mock(side_effect=Exception("Chart error"))
        mock_st.pyplot.side_effect = Exception("Chart error")
        
        _render_chart(mock_fig)
        
        mock_st.error.assert_called_with("Failed to render chart: Chart error")
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface.has_dataframe')
    def test_handle_user_input_no_data(self, mock_has_dataframe, mock_st):
        """Test handling user input when no data is loaded."""
        mock_has_dataframe.return_value = False
        
        result = handle_user_input()
        
        mock_st.warning.assert_called_with("Please upload a dataset before asking questions.")
        assert result is None
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface.has_dataframe')
    @patch('components.chat_interface.add_message')
    def test_handle_user_input_with_query(self, mock_add_message, mock_has_dataframe, mock_st):
        """Test handling user input with a valid query."""
        mock_has_dataframe.return_value = True
        mock_st.chat_input.return_value = "What is the mean of column A?"
        
        result = handle_user_input()
        
        mock_add_message.assert_called_with(
            role="user",
            content="What is the mean of column A?",
            message_type="text"
        )
        assert result == "What is the mean of column A?"
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface.has_dataframe')
    def test_handle_user_input_empty_query(self, mock_has_dataframe, mock_st):
        """Test handling empty user input."""
        mock_has_dataframe.return_value = True
        mock_st.chat_input.return_value = None
        
        result = handle_user_input()
        
        assert result is None
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface.add_message')
    def test_display_response(self, mock_add_message, mock_st):
        """Test displaying a response."""
        display_response("Test response", "text")
        
        mock_add_message.assert_called_with(
            role="agent",
            content="Test response",
            message_type="text",
            chart_data=None
        )
        mock_st.rerun.assert_called_once()
    
    @patch('components.chat_interface.st')
    @patch('components.chat_interface.add_message')
    def test_display_chart_response(self, mock_add_message, mock_st):
        """Test displaying a chart response."""
        mock_chart = Mock()
        
        display_chart_response("Here's your chart", mock_chart)
        
        mock_add_message.assert_called_with(
            role="agent",
            content="Here's your chart",
            message_type="plot",
            chart_data=mock_chart
        )
        mock_st.rerun.assert_called_once()
    
    @patch('components.chat_interface.display_response')
    def test_display_error_message(self, mock_display_response):
        """Test displaying an error message."""
        display_error_message("Something went wrong")
        
        mock_display_response.assert_called_with("Something went wrong", message_type="error")
    
    @patch('components.chat_interface.display_response')
    def test_display_text_response(self, mock_display_response):
        """Test displaying a text response."""
        display_text_response("Here's your answer")
        
        mock_display_response.assert_called_with("Here's your answer", message_type="text")
    
    @patch('components.chat_interface.st')
    def test_clear_chat(self, mock_st):
        """Test clearing chat history."""
        with patch('utils.session_manager.clear_chat_history') as mock_clear:
            clear_chat()
            
            mock_clear.assert_called_once()
            mock_st.rerun.assert_called_once()
    
    @patch('components.chat_interface.get_chat_history')
    def test_get_chat_stats_empty(self, mock_get_history):
        """Test getting chat statistics with empty history."""
        mock_get_history.return_value = []
        
        stats = get_chat_stats()
        
        expected = {
            "total_messages": 0,
            "user_messages": 0,
            "agent_messages": 0,
            "error_messages": 0,
            "plot_messages": 0,
            "session_start": None,
            "last_message": None
        }
        
        assert stats == expected
    
    @patch('components.chat_interface.get_chat_history')
    def test_get_chat_stats_with_messages(self, mock_get_history):
        """Test getting chat statistics with messages."""
        start_time = datetime(2024, 1, 1, 12, 0, 0)
        end_time = datetime(2024, 1, 1, 12, 5, 0)
        
        test_messages = [
            ChatMessage("user", "Question 1", start_time, "text"),
            ChatMessage("agent", "Answer 1", datetime(2024, 1, 1, 12, 1, 0), "text"),
            ChatMessage("user", "Question 2", datetime(2024, 1, 1, 12, 2, 0), "text"),
            ChatMessage("agent", "Chart response", datetime(2024, 1, 1, 12, 3, 0), "plot"),
            ChatMessage("agent", "Error occurred", end_time, "error")
        ]
        
        mock_get_history.return_value = test_messages
        
        stats = get_chat_stats()
        
        expected = {
            "total_messages": 5,
            "user_messages": 2,
            "agent_messages": 3,
            "error_messages": 1,
            "plot_messages": 1,
            "session_start": start_time,
            "last_message": end_time
        }
        
        assert stats == expected


class TestChatInterfaceIntegration:
    """Integration tests for chat interface with session manager."""
    
    def setup_method(self):
        """Set up test environment."""
        clear_session()
        initialize_session()
    
    def teardown_method(self):
        """Clean up after tests."""
        clear_session()
    
    @patch('components.chat_interface.st')
    def test_full_conversation_flow(self, mock_st):
        """Test a complete conversation flow."""
        from utils.session_manager import add_message, get_chat_history
        
        # Simulate a conversation
        add_message("user", "What is the average age?", "text")
        add_message("agent", "The average age is 35.2 years", "text")
        add_message("user", "Show me a chart", "text")
        
        # Mock chart data
        mock_chart = Mock()
        add_message("agent", "Here's your chart", "plot", mock_chart)
        
        # Get chat history and verify
        history = get_chat_history()
        assert len(history) == 4
        
        # Verify message types and content
        assert history[0].role == "user"
        assert history[0].content == "What is the average age?"
        assert history[1].role == "agent"
        assert history[1].content == "The average age is 35.2 years"
        assert history[2].role == "user"
        assert history[3].role == "agent"
        assert history[3].message_type == "plot"
        assert history[3].chart_data == mock_chart
    
    @patch('components.chat_interface.st')
    def test_error_handling_in_conversation(self, mock_st):
        """Test error handling during conversation."""
        from utils.session_manager import add_message, get_chat_history
        
        # Add normal message
        add_message("user", "Invalid query", "text")
        
        # Add error response
        add_message("agent", "Unable to process query", "error")
        
        history = get_chat_history()
        assert len(history) == 2
        assert history[1].message_type == "error"
        
        # Test stats
        stats = get_chat_stats()
        assert stats["error_messages"] == 1
        assert stats["total_messages"] == 2


if __name__ == "__main__":
    pytest.main([__file__])