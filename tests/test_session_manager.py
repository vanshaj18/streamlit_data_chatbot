"""
Unit tests for session state management.
"""

import pytest
import pandas as pd
from datetime import datetime
from unittest.mock import Mock, patch
import sys
import os

# Add the parent directory to the path to import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.session_manager import (
    ChatMessage, FileInfo, SessionState,
    initialize_session, get_session_data, update_dataframe,
    add_message, get_chat_history, has_dataframe, get_dataframe,
    get_file_info, set_agent, get_agent, clear_session,
    clear_chat_history, get_session_summary
)


class TestChatMessage:
    """Test ChatMessage data class."""
    
    def test_chat_message_creation(self):
        """Test creating a ChatMessage instance."""
        timestamp = datetime.now()
        message = ChatMessage(
            role="user",
            content="Hello",
            timestamp=timestamp,
            message_type="text"
        )
        
        assert message.role == "user"
        assert message.content == "Hello"
        assert message.timestamp == timestamp
        assert message.message_type == "text"
        assert message.chart_data is None
    
    def test_chat_message_with_chart_data(self):
        """Test ChatMessage with chart data."""
        chart_data = {"type": "bar", "data": [1, 2, 3]}
        message = ChatMessage(
            role="agent",
            content="Here's your chart",
            timestamp=datetime.now(),
            message_type="plot",
            chart_data=chart_data
        )
        
        assert message.chart_data == chart_data
        assert message.message_type == "plot"


class TestFileInfo:
    """Test FileInfo data class."""
    
    def test_file_info_creation(self):
        """Test creating a FileInfo instance."""
        timestamp = datetime.now()
        file_info = FileInfo(
            filename="test.csv",
            file_size=1024,
            upload_timestamp=timestamp,
            columns=["col1", "col2"],
            row_count=100,
            file_type="csv"
        )
        
        assert file_info.filename == "test.csv"
        assert file_info.file_size == 1024
        assert file_info.upload_timestamp == timestamp
        assert file_info.columns == ["col1", "col2"]
        assert file_info.row_count == 100
        assert file_info.file_type == "csv"


class TestSessionState:
    """Test SessionState data class."""
    
    def test_session_state_default_values(self):
        """Test SessionState default initialization."""
        state = SessionState()
        
        assert state.dataframe is None
        assert state.chat_history == []
        assert state.agent is None
        assert state.file_info is None
        assert state.is_initialized is False
    
    def test_session_state_with_data(self):
        """Test SessionState with data."""
        df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
        messages = [ChatMessage("user", "test", datetime.now(), "text")]
        
        state = SessionState(
            dataframe=df,
            chat_history=messages,
            is_initialized=True
        )
        
        assert state.dataframe is not None
        assert len(state.chat_history) == 1
        assert state.is_initialized is True


class TestSessionManager:
    """Test session manager functions."""
    
    def setup_method(self):
        """Set up test environment before each test."""
        # Mock streamlit session state as an object with attributes
        class MockSessionState:
            def __init__(self):
                self._data = {}
            
            def __contains__(self, key):
                return key in self._data
            
            def __getattr__(self, key):
                if key.startswith('_'):
                    return object.__getattribute__(self, key)
                return self._data.get(key)
            
            def __setattr__(self, key, value):
                if key.startswith('_'):
                    object.__setattr__(self, key, value)
                else:
                    self._data[key] = value
            
            def __delattr__(self, key):
                if key in self._data:
                    del self._data[key]
        
        self.mock_session_state = MockSessionState()
        self.patcher = patch('utils.session_manager.st.session_state', self.mock_session_state)
        self.patcher.start()
    
    def teardown_method(self):
        """Clean up after each test."""
        self.patcher.stop()
    
    def test_initialize_session(self):
        """Test session initialization."""
        initialize_session()
        
        assert 'session_data' in self.mock_session_state
        assert isinstance(self.mock_session_state.session_data, SessionState)
        assert self.mock_session_state.session_data.is_initialized is True
    
    def test_initialize_session_already_exists(self):
        """Test that initialization doesn't overwrite existing session."""
        # Create initial session
        initialize_session()
        original_session = self.mock_session_state.session_data
        
        # Try to initialize again
        initialize_session()
        
        # Should be the same instance
        assert self.mock_session_state.session_data is original_session
    
    def test_get_session_data(self):
        """Test getting session data."""
        session_data = get_session_data()
        
        assert isinstance(session_data, SessionState)
        assert session_data.is_initialized is True
    
    def test_get_session_data_auto_initialize(self):
        """Test that get_session_data initializes if needed."""
        # Don't initialize first
        session_data = get_session_data()
        
        assert 'session_data' in self.mock_session_state
        assert isinstance(session_data, SessionState)
    
    def test_update_dataframe(self):
        """Test updating DataFrame in session."""
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        file_info = {
            'filename': 'test.csv',
            'file_size': 1024,
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(df, file_info)
        session_data = get_session_data()
        
        assert session_data.dataframe is not None
        assert session_data.dataframe.equals(df)
        assert session_data.file_info.filename == 'test.csv'
        assert session_data.file_info.row_count == 3
        assert session_data.file_info.columns == ['A', 'B']
        assert session_data.agent is None  # Should reset agent
    
    def test_add_message(self):
        """Test adding messages to chat history."""
        add_message("user", "Hello", "text")
        add_message("agent", "Hi there", "text")
        
        session_data = get_session_data()
        assert len(session_data.chat_history) == 2
        
        first_message = session_data.chat_history[0]
        assert first_message.role == "user"
        assert first_message.content == "Hello"
        assert first_message.message_type == "text"
        
        second_message = session_data.chat_history[1]
        assert second_message.role == "agent"
        assert second_message.content == "Hi there"
    
    def test_add_message_with_chart_data(self):
        """Test adding message with chart data."""
        chart_data = {"type": "bar", "values": [1, 2, 3]}
        add_message("agent", "Here's your chart", "plot", chart_data)
        
        session_data = get_session_data()
        message = session_data.chat_history[0]
        
        assert message.message_type == "plot"
        assert message.chart_data == chart_data
    
    def test_get_chat_history(self):
        """Test getting chat history."""
        add_message("user", "Test message", "text")
        
        history = get_chat_history()
        assert len(history) == 1
        assert history[0].content == "Test message"
    
    def test_has_dataframe(self):
        """Test checking if DataFrame exists."""
        # Initially no DataFrame
        assert has_dataframe() is False
        
        # Add DataFrame
        df = pd.DataFrame({"A": [1, 2]})
        file_info = {'filename': 'test.csv', 'file_size': 100, 'file_type': 'csv'}
        update_dataframe(df, file_info)
        
        assert has_dataframe() is True
    
    def test_get_dataframe(self):
        """Test getting DataFrame from session."""
        # Initially None
        assert get_dataframe() is None
        
        # Add DataFrame
        df = pd.DataFrame({"A": [1, 2, 3]})
        file_info = {'filename': 'test.csv', 'file_size': 100, 'file_type': 'csv'}
        update_dataframe(df, file_info)
        
        retrieved_df = get_dataframe()
        assert retrieved_df is not None
        assert retrieved_df.equals(df)
    
    def test_get_file_info(self):
        """Test getting file information."""
        # Initially None
        assert get_file_info() is None
        
        # Add file info
        df = pd.DataFrame({"A": [1, 2]})
        file_info = {
            'filename': 'test.csv',
            'file_size': 1024,
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        update_dataframe(df, file_info)
        
        retrieved_info = get_file_info()
        assert retrieved_info is not None
        assert retrieved_info.filename == 'test.csv'
        assert retrieved_info.file_size == 1024
    
    def test_set_and_get_agent(self):
        """Test setting and getting agent."""
        # Initially None
        assert get_agent() is None
        
        # Set agent
        mock_agent = Mock()
        set_agent(mock_agent)
        
        assert get_agent() is mock_agent
    
    def test_clear_session(self):
        """Test clearing session."""
        # Add some data
        df = pd.DataFrame({"A": [1, 2]})
        file_info = {'filename': 'test.csv', 'file_size': 100, 'file_type': 'csv'}
        update_dataframe(df, file_info)
        add_message("user", "Test", "text")
        set_agent(Mock())
        
        # Verify data exists
        assert has_dataframe() is True
        assert len(get_chat_history()) == 1
        assert get_agent() is not None
        
        # Clear session
        clear_session()
        
        # Verify data is cleared
        assert has_dataframe() is False
        assert len(get_chat_history()) == 0
        assert get_agent() is None
        assert get_session_data().is_initialized is True  # Should be re-initialized
    
    def test_clear_chat_history(self):
        """Test clearing only chat history."""
        # Add data
        df = pd.DataFrame({"A": [1, 2]})
        file_info = {'filename': 'test.csv', 'file_size': 100, 'file_type': 'csv'}
        update_dataframe(df, file_info)
        add_message("user", "Test", "text")
        set_agent(Mock())
        
        # Clear only chat history
        clear_chat_history()
        
        # Chat history should be empty, other data should remain
        assert len(get_chat_history()) == 0
        assert has_dataframe() is True
        assert get_agent() is not None
    
    def test_get_session_summary(self):
        """Test getting session summary."""
        # Empty session
        summary = get_session_summary()
        assert summary['has_dataframe'] is False
        assert summary['dataframe_shape'] is None
        assert summary['chat_message_count'] == 0
        assert summary['has_agent'] is False
        assert summary['file_info'] is None
        assert summary['is_initialized'] is True
        
        # Add data
        df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
        file_info = {
            'filename': 'test.csv',
            'file_size': 1024,
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        update_dataframe(df, file_info)
        add_message("user", "Test", "text")
        set_agent(Mock())
        
        # Get summary with data
        summary = get_session_summary()
        assert summary['has_dataframe'] is True
        assert summary['dataframe_shape'] == (3, 2)
        assert summary['chat_message_count'] == 1
        assert summary['has_agent'] is True
        assert summary['file_info']['filename'] == 'test.csv'
        assert summary['file_info']['row_count'] == 3
        assert summary['file_info']['column_count'] == 2


if __name__ == "__main__":
    pytest.main([__file__])