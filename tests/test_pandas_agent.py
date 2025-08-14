"""
Unit tests for PandasAI agent wrapper.
"""

import pytest
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from services.pandas_agent import PandasAgent, AgentResponse


class TestPandasAgent:
    """Test cases for PandasAgent class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.sample_df = pd.DataFrame({
            'name': ['Alice', 'Bob', 'Charlie'],
            'age': [25, 30, 35],
            'salary': [50000, 60000, 70000]
        })
        
        # Mock API key for testing
        self.test_api_key = "test-api-key-123"
    
    def test_init_without_api_key(self):
        """Test initialization without API key."""
        with patch.dict(os.environ, {}, clear=True):
            agent = PandasAgent()
            assert agent.api_key is None
            assert agent.agent is None
            assert agent.dataframe is None
    
    def test_init_with_api_key(self):
        """Test initialization with API key."""
        agent = PandasAgent(api_key=self.test_api_key)
        assert agent.api_key == self.test_api_key
        assert agent.agent is None
        assert agent.dataframe is None
    
    def test_init_with_env_api_key(self):
        """Test initialization with API key from environment."""
        with patch.dict(os.environ, {'GEMINI_API_KEY': self.test_api_key}):
            agent = PandasAgent()
            assert agent.api_key == self.test_api_key
    
    @patch('services.pandas_agent.PANDASAI_AVAILABLE', False)
    def test_initialize_agent_pandasai_not_available(self):
        """Test agent initialization when PandasAI is not available."""
        agent = PandasAgent(api_key=self.test_api_key)
        result = agent.initialize_agent(self.sample_df)
        assert result is False
    
    @patch('services.pandas_agent.PANDASAI_AVAILABLE', True)
    def test_initialize_agent_no_api_key(self):
        """Test agent initialization without API key."""
        agent = PandasAgent()
        result = agent.initialize_agent(self.sample_df)
        assert result is False
    
    @patch('services.pandas_agent.PANDASAI_AVAILABLE', True)
    def test_initialize_agent_empty_dataframe(self):
        """Test agent initialization with empty DataFrame."""
        agent = PandasAgent(api_key=self.test_api_key)
        empty_df = pd.DataFrame()
        result = agent.initialize_agent(empty_df)
        assert result is False
    
    @patch('services.pandas_agent.PANDASAI_AVAILABLE', True)
    @patch('services.pandas_agent.GooglePalm')
    @patch('services.pandas_agent.Agent')
    def test_initialize_agent_success(self, mock_agent_class, mock_GooglePalm_class):
        """Test successful agent initialization."""
        # Mock the GooglePalm LLM
        mock_llm = Mock()
        mock_GooglePalm_class.return_value = mock_llm
        
        # Mock the Agent
        mock_agent = Mock()
        mock_agent_class.return_value = mock_agent
        
        agent = PandasAgent(api_key=self.test_api_key)
        result = agent.initialize_agent(self.sample_df)
        
        assert result is True
        assert agent.agent == mock_agent
        assert agent.dataframe.equals(self.sample_df)
        
        # Verify GooglePalm was initialized with correct API key
        mock_GooglePalm_class.assert_called_once_with(api_token=self.test_api_key)
        
        # Verify Agent was initialized with DataFrame and config
        mock_agent_class.assert_called_once()
        call_args = mock_agent_class.call_args
        assert call_args[0][0].equals(self.sample_df)  # DataFrame argument
        assert 'config' in call_args[1]  # Config argument
    
    @patch('services.pandas_agent.PANDASAI_AVAILABLE', True)
    @patch('services.pandas_agent.GooglePalm')
    @patch('services.pandas_agent.Agent')
    def test_initialize_agent_exception(self, mock_agent_class, mock_GooglePalm_class):
        """Test agent initialization with exception."""
        mock_GooglePalm_class.side_effect = Exception("API error")
        
        agent = PandasAgent(api_key=self.test_api_key)
        result = agent.initialize_agent(self.sample_df)
        
        assert result is False
        assert agent.agent is None
    
    def test_process_query_agent_not_initialized(self):
        """Test query processing when agent is not initialized."""
        agent = PandasAgent(api_key=self.test_api_key)
        response = agent.process_query("What is the average age?")
        
        assert response.response_type == "error"
        assert "not initialized" in response.error_message.lower()
        assert response.content is None
    
    def test_process_query_empty_query(self):
        """Test query processing with empty query."""
        agent = PandasAgent(api_key=self.test_api_key)
        agent.agent = Mock()  # Mock initialized agent
        
        response = agent.process_query("")
        
        assert response.response_type == "error"
        assert "empty" in response.error_message.lower()
        assert response.content is None
    
    def test_process_query_whitespace_only(self):
        """Test query processing with whitespace-only query."""
        agent = PandasAgent(api_key=self.test_api_key)
        agent.agent = Mock()  # Mock initialized agent
        
        response = agent.process_query("   \n\t   ")
        
        assert response.response_type == "error"
        assert "empty" in response.error_message.lower()
    
    def test_process_query_success_text_response(self):
        """Test successful query processing with text response."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        # Mock the agent
        mock_agent = Mock()
        mock_agent.chat.return_value = "The average age is 30 years."
        agent.agent = mock_agent
        
        response = agent.process_query("What is the average age?")
        
        assert response.response_type == "text"
        assert response.content == "The average age is 30 years."
        assert response.error_message is None
        assert response.execution_time >= 0
        
        # Verify the agent was called with correct query
        mock_agent.chat.assert_called_once_with("What is the average age?")
    
    def test_process_query_success_dataframe_response(self):
        """Test successful query processing with DataFrame response."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        # Mock the agent to return a DataFrame
        mock_agent = Mock()
        result_df = pd.DataFrame({'result': [1, 2, 3]})
        mock_agent.chat.return_value = result_df
        agent.agent = mock_agent
        
        response = agent.process_query("Show me the data")
        
        assert response.response_type == "dataframe"
        assert response.content.equals(result_df)
        assert response.error_message is None
    
    def test_process_query_exception(self):
        """Test query processing with exception."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        # Mock the agent to raise an exception
        mock_agent = Mock()
        mock_agent.chat.side_effect = Exception("API timeout")
        agent.agent = mock_agent
        
        response = agent.process_query("What is the average age?")
        
        assert response.response_type == "error"
        assert response.content is None
        assert "too long" in response.error_message.lower()  # The error formatter converts "timeout" to "too long"
    
    def test_handle_response_none(self):
        """Test handling None response."""
        agent = PandasAgent(api_key=self.test_api_key)
        timestamp = datetime.now()
        
        response = agent.handle_response(None, 1.5, timestamp)
        
        assert response.response_type == "text"
        assert "no response" in response.content.lower()
        assert response.execution_time == 1.5
        assert response.timestamp == timestamp
    
    def test_handle_response_dataframe(self):
        """Test handling DataFrame response."""
        agent = PandasAgent(api_key=self.test_api_key)
        timestamp = datetime.now()
        df_response = pd.DataFrame({'col': [1, 2, 3]})
        
        response = agent.handle_response(df_response, 2.0, timestamp)
        
        assert response.response_type == "dataframe"
        assert response.content.equals(df_response)
        assert response.execution_time == 2.0
    
    def test_handle_response_text(self):
        """Test handling text response."""
        agent = PandasAgent(api_key=self.test_api_key)
        timestamp = datetime.now()
        text_response = "This is a text answer"
        
        response = agent.handle_response(text_response, 1.0, timestamp)
        
        assert response.response_type == "text"
        assert response.content == text_response
        assert response.execution_time == 1.0
    
    def test_handle_response_exception(self):
        """Test handling response with exception."""
        agent = PandasAgent(api_key=self.test_api_key)
        timestamp = datetime.now()
        
        # Create a mock object that raises exception during processing
        class BadResponse:
            def __str__(self):
                raise Exception("Conversion error")
        
        mock_response = BadResponse()
        
        response = agent.handle_response(mock_response, 1.0, timestamp)
        
        assert response.response_type == "error"
        assert response.content is None
        assert "error processing response" in response.error_message.lower()
    
    def test_format_error_message_api_key(self):
        """Test error message formatting for API key issues."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        error_msg = agent._format_error_message("Invalid API key provided")
        assert "api key" in error_msg.lower()
    
    def test_format_error_message_timeout(self):
        """Test error message formatting for timeout issues."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        error_msg = agent._format_error_message("Request timeout occurred")
        assert "too long" in error_msg.lower()
    
    def test_format_error_message_rate_limit(self):
        """Test error message formatting for rate limit issues."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        error_msg = agent._format_error_message("Rate limit exceeded")
        assert "too many requests" in error_msg.lower()
    
    def test_format_error_message_column_not_found(self):
        """Test error message formatting for column not found."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        error_msg = agent._format_error_message("Column 'xyz' not found in DataFrame")
        assert "column not found" in error_msg.lower()
    
    def test_format_error_message_generic(self):
        """Test error message formatting for generic errors."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        error_msg = agent._format_error_message("Some unknown error occurred")
        assert "unable to process" in error_msg.lower()
    
    def test_get_dataframe_info_no_dataframe(self):
        """Test getting DataFrame info when no DataFrame is loaded."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        info = agent.get_dataframe_info()
        assert info is None
    
    def test_get_dataframe_info_success(self):
        """Test getting DataFrame info successfully."""
        agent = PandasAgent(api_key=self.test_api_key)
        agent.dataframe = self.sample_df
        
        info = agent.get_dataframe_info()
        
        assert info is not None
        assert info['shape'] == (3, 3)
        assert set(info['columns']) == {'name', 'age', 'salary'}
        assert 'dtypes' in info
        assert 'memory_usage' in info
        assert 'null_counts' in info
    
    def test_get_dataframe_info_exception(self):
        """Test getting DataFrame info with exception."""
        agent = PandasAgent(api_key=self.test_api_key)
        
        # Create a mock DataFrame that raises exception
        mock_df = Mock()
        mock_df.shape.side_effect = Exception("Access error")
        agent.dataframe = mock_df
        
        info = agent.get_dataframe_info()
        assert info is None
    
    @patch('services.pandas_agent.PANDASAI_AVAILABLE', True)
    def test_is_initialized_true(self):
        """Test is_initialized returns True when properly initialized."""
        agent = PandasAgent(api_key=self.test_api_key)
        agent.agent = Mock()
        agent.dataframe = self.sample_df
        
        assert agent.is_initialized() is True
    
    @patch('services.pandas_agent.PANDASAI_AVAILABLE', False)
    def test_is_initialized_pandasai_not_available(self):
        """Test is_initialized returns False when PandasAI not available."""
        agent = PandasAgent(api_key=self.test_api_key)
        agent.agent = Mock()
        agent.dataframe = self.sample_df
        
        assert agent.is_initialized() is False
    
    def test_is_initialized_no_agent(self):
        """Test is_initialized returns False when agent not set."""
        agent = PandasAgent(api_key=self.test_api_key)
        agent.dataframe = self.sample_df
        
        assert agent.is_initialized() is False
    
    def test_is_initialized_no_dataframe(self):
        """Test is_initialized returns False when dataframe not set."""
        agent = PandasAgent(api_key=self.test_api_key)
        agent.agent = Mock()
        
        assert agent.is_initialized() is False
    
    def test_is_initialized_no_api_key(self):
        """Test is_initialized returns False when API key not set."""
        agent = PandasAgent()
        agent.agent = Mock()
        agent.dataframe = self.sample_df
        
        assert agent.is_initialized() is False


class TestAgentResponse:
    """Test cases for AgentResponse dataclass."""
    
    def test_agent_response_creation(self):
        """Test creating AgentResponse instance."""
        timestamp = datetime.now()
        response = AgentResponse(
            content="Test content",
            response_type="text",
            execution_time=1.5,
            timestamp=timestamp,
            error_message=None
        )
        
        assert response.content == "Test content"
        assert response.response_type == "text"
        assert response.execution_time == 1.5
        assert response.timestamp == timestamp
        assert response.error_message is None
    
    def test_agent_response_with_error(self):
        """Test creating AgentResponse with error."""
        timestamp = datetime.now()
        response = AgentResponse(
            content=None,
            response_type="error",
            execution_time=0.5,
            timestamp=timestamp,
            error_message="Test error"
        )
        
        assert response.content is None
        assert response.response_type == "error"
        assert response.error_message == "Test error"


if __name__ == "__main__":
    pytest.main([__file__])