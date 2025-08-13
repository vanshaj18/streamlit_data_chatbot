"""
Integration tests for error handling across components.
"""

import pytest
import pandas as pd
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
import io
from datetime import datetime

from components.file_handler import validate_file, load_dataframe
from services.pandas_agent import PandasAgent, AgentResponse
from components.visualization import render_chart, handle_chart_error
from utils.error_handler import ErrorCategory, ErrorSeverity


class TestFileHandlerErrorIntegration:
    """Integration tests for file handler error scenarios."""
    
    def test_file_size_validation_error(self):
        """Test file size validation with oversized file."""
        # Create mock uploaded file that's too large
        mock_file = Mock()
        mock_file.name = "large_file.csv"
        mock_file.size = 60 * 1024 * 1024  # 60MB (over 50MB limit)
        
        with patch('streamlit.error') as mock_st_error:
            is_valid, error_message = validate_file(mock_file)
        
        assert not is_valid
        assert "file is too large" in error_message.lower()
        # Error should not be displayed in UI during validation (show_ui=False)
        mock_st_error.assert_not_called()
    
    def test_unsupported_file_format_error(self):
        """Test validation with unsupported file format."""
        mock_file = Mock()
        mock_file.name = "document.pdf"
        mock_file.size = 1024 * 1024  # 1MB
        
        with patch('streamlit.error') as mock_st_error:
            is_valid, error_message = validate_file(mock_file)
        
        assert not is_valid
        assert "not supported" in error_message.lower()
        mock_st_error.assert_not_called()
    
    def test_corrupted_csv_file_error(self):
        """Test loading corrupted CSV file."""
        # Create mock file that will cause pandas to fail
        mock_file = Mock()
        mock_file.name = "corrupted.csv"
        mock_file.size = 1024
        mock_file.seek = Mock()
        
        # Mock pandas read_csv to raise an exception
        with patch('pandas.read_csv', side_effect=pd.errors.EmptyDataError("No data")):
            with patch('streamlit.error') as mock_st_error:
                df, error_message = load_dataframe(mock_file)
        
        assert df is None
        assert error_message is not None
        assert len(error_message) > 0
        mock_st_error.assert_not_called()  # show_ui=False in load_dataframe
    
    def test_empty_dataframe_error(self):
        """Test loading file that results in empty DataFrame."""
        mock_file = Mock()
        mock_file.name = "empty.csv"
        mock_file.size = 100
        mock_file.seek = Mock()
        
        # Mock pandas to return empty DataFrame
        empty_df = pd.DataFrame()
        with patch('pandas.read_csv', return_value=empty_df):
            with patch('streamlit.error') as mock_st_error:
                df, error_message = load_dataframe(mock_file)
        
        assert df is None
        assert "empty" in error_message.lower()
        mock_st_error.assert_not_called()
    
    def test_too_many_columns_error(self):
        """Test loading file with too many columns."""
        mock_file = Mock()
        mock_file.name = "wide_data.csv"
        mock_file.size = 1024
        mock_file.seek = Mock()
        
        # Create DataFrame with too many columns
        wide_df = pd.DataFrame({f'col_{i}': [1, 2, 3] for i in range(1001)})
        
        with patch('pandas.read_csv', return_value=wide_df):
            with patch('streamlit.error') as mock_st_error:
                df, error_message = load_dataframe(mock_file)
        
        assert df is None
        assert "too many columns" in error_message.lower()
        mock_st_error.assert_not_called()


class TestPandasAgentErrorIntegration:
    """Integration tests for PandasAI agent error scenarios."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.agent = PandasAgent()
        self.sample_df = pd.DataFrame({
            'A': [1, 2, 3],
            'B': ['x', 'y', 'z']
        })
    
    def test_uninitialized_agent_error(self):
        """Test querying uninitialized agent."""
        with patch('streamlit.error') as mock_st_error:
            response = self.agent.process_query("Show me data")
        
        assert response.response_type == "error"
        assert "not initialized" in response.error_message.lower()
        mock_st_error.assert_not_called()  # show_ui=False in process_query
    
    def test_empty_query_error(self):
        """Test processing empty query."""
        # Initialize agent first
        with patch('pandasai.Agent'):
            self.agent.initialize_agent(self.sample_df)
        
        with patch('streamlit.error') as mock_st_error:
            response = self.agent.process_query("")
        
        assert response.response_type == "error"
        assert "empty" in response.error_message.lower()
        mock_st_error.assert_not_called()
    
    @patch('pandasai.Agent')
    def test_api_key_error(self, mock_agent_class):
        """Test API key related errors."""
        # Mock agent initialization to fail with API key error
        mock_agent_class.side_effect = Exception("API key invalid")
        
        with patch('streamlit.error') as mock_st_error:
            success = self.agent.initialize_agent(self.sample_df)
        
        assert not success
        mock_st_error.assert_not_called()
    
    @patch('pandasai.Agent')
    def test_query_timeout_error(self, mock_agent_class):
        """Test query timeout handling."""
        # Mock agent to raise timeout error
        mock_agent_instance = Mock()
        mock_agent_instance.chat.side_effect = TimeoutError("Query timeout")
        mock_agent_class.return_value = mock_agent_instance
        
        self.agent.initialize_agent(self.sample_df)
        
        with patch('streamlit.error') as mock_st_error:
            response = self.agent.process_query("Complex query")
        
        assert response.response_type == "error"
        assert "too long" in response.error_message.lower()
        mock_st_error.assert_not_called()


class TestVisualizationErrorIntegration:
    """Integration tests for visualization error scenarios."""
    
    def test_none_chart_data_error(self):
        """Test rendering None chart data."""
        with patch('streamlit.error') as mock_st_error:
            with patch('streamlit.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock()
                mock_expander.return_value.__exit__ = Mock(return_value=None)
                
                success = render_chart(None, "Test Chart")
        
        assert not success
        mock_st_error.assert_called_once()
    
    def test_invalid_matplotlib_figure_error(self):
        """Test rendering invalid matplotlib figure."""
        # Create invalid figure object
        invalid_figure = "not a figure"
        
        with patch('streamlit.error') as mock_st_error:
            with patch('streamlit.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock()
                mock_expander.return_value.__exit__ = Mock(return_value=None)
                
                success = render_chart(invalid_figure, "Test Chart")
        
        assert not success
        mock_st_error.assert_called_once()
    
    def test_chart_rendering_exception(self):
        """Test handling of exceptions during chart rendering."""
        # Mock matplotlib figure that raises exception
        mock_figure = Mock()
        mock_figure.savefig.side_effect = Exception("Rendering failed")
        
        with patch('streamlit.pyplot', side_effect=Exception("Streamlit error")):
            with patch('streamlit.error') as mock_st_error:
                with patch('streamlit.expander') as mock_expander:
                    mock_expander.return_value.__enter__ = Mock()
                    mock_expander.return_value.__exit__ = Mock(return_value=None)
                    
                    success = render_chart(mock_figure, "Test Chart")
        
        assert not success
        mock_st_error.assert_called_once()


class TestEndToEndErrorScenarios:
    """End-to-end error scenario tests."""
    
    def test_complete_file_upload_error_flow(self):
        """Test complete error flow from file upload to UI display."""
        # Create oversized file
        mock_file = Mock()
        mock_file.name = "huge_file.csv"
        mock_file.size = 100 * 1024 * 1024  # 100MB
        
        # Test validation
        with patch('streamlit.error') as mock_st_error:
            is_valid, error_message = validate_file(mock_file)
        
        assert not is_valid
        assert "too large" in error_message.lower()
        
        # Simulate UI display of error
        with patch('streamlit.error') as mock_st_error:
            from utils.error_handler import handle_file_error
            error_info = handle_file_error(error_message, mock_file.name, show_ui=True)
        
        assert error_info.category == ErrorCategory.FILE_UPLOAD
        assert error_info.severity in [ErrorSeverity.MEDIUM, ErrorSeverity.HIGH]
        mock_st_error.assert_called_once()
    
    def test_query_processing_error_chain(self):
        """Test error propagation through query processing chain."""
        # Start with uninitialized agent
        agent = PandasAgent()
        
        # Process query (should fail)
        with patch('streamlit.error') as mock_st_error:
            response = agent.process_query("Show me data")
        
        assert response.response_type == "error"
        
        # Simulate chat interface handling the error
        from components.chat_interface import display_error_message
        
        with patch('streamlit.error') as mock_st_error:
            with patch('utils.session_manager.add_message') as mock_add_message:
                with patch('streamlit.rerun') as mock_rerun:
                    display_error_message(response.error_message)
        
        mock_add_message.assert_called_once()
        mock_rerun.assert_called_once()
    
    def test_visualization_error_recovery(self):
        """Test visualization error with recovery suggestions."""
        error_message = "Memory error during chart creation"
        
        with patch('streamlit.error') as mock_st_error:
            with patch('streamlit.expander') as mock_expander:
                mock_expander_context = MagicMock()
                mock_expander.return_value.__enter__ = Mock(return_value=mock_expander_context)
                mock_expander.return_value.__exit__ = Mock(return_value=None)
                
                success = handle_chart_error(error_message)
        
        assert not success
        mock_st_error.assert_called_once()
        mock_expander.assert_called_once()
        
        # Check that recovery suggestions were provided
        error_call_args = mock_st_error.call_args[0][0]
        assert "error" in error_call_args.lower()


class TestErrorLogging:
    """Test error logging functionality."""
    
    def test_error_logging_levels(self):
        """Test that errors are logged at appropriate levels."""
        from utils.error_handler import ErrorHandler
        
        handler = ErrorHandler()
        
        with patch.object(handler.logger, 'critical') as mock_critical:
            with patch.object(handler.logger, 'error') as mock_error:
                with patch.object(handler.logger, 'warning') as mock_warning:
                    with patch.object(handler.logger, 'info') as mock_info:
                        # Test critical error logging
                        handler.handle_error(
                            "Out of memory error",
                            ErrorCategory.SYSTEM_ERROR,
                            show_ui=False
                        )
                        mock_critical.assert_called_once()
                        
                        # Test high severity error logging
                        handler.handle_error(
                            "API key authentication failed",
                            ErrorCategory.API_ERROR,
                            show_ui=False
                        )
                        mock_error.assert_called_once()
                        
                        # Test medium severity error logging
                        handler.handle_error(
                            "Connection timeout",
                            ErrorCategory.QUERY_PROCESSING,
                            show_ui=False
                        )
                        mock_warning.assert_called_once()
                        
                        # Test low severity error logging
                        handler.handle_error(
                            "Chart rendering issue",
                            ErrorCategory.VISUALIZATION,
                            show_ui=False
                        )
                        mock_info.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])