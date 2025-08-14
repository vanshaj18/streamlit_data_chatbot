"""
Unit tests for the centralized error handling system.
"""

import pytest
import streamlit as st
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from utils.error_handler import (
    ErrorHandler, ErrorCategory, ErrorSeverity, ErrorInfo,
    handle_file_error, handle_query_error, handle_viz_error, safe_execute
)


class TestErrorHandler:
    """Test cases for the ErrorHandler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_error_handler_initialization(self):
        """Test that error handler initializes correctly."""
        assert self.error_handler is not None
        assert hasattr(self.error_handler, '_error_patterns')
        assert hasattr(self.error_handler, '_recovery_suggestions')
        assert len(self.error_handler._error_patterns) > 0
        assert len(self.error_handler._recovery_suggestions) > 0
    
    def test_handle_error_with_exception(self):
        """Test handling of Exception objects."""
        test_exception = ValueError("Test error message")
        
        with patch('streamlit.error') as mock_st_error:
            error_info = self.error_handler.handle_error(
                test_exception, 
                ErrorCategory.FILE_UPLOAD,
                show_ui=True
            )
        
        assert error_info.category == ErrorCategory.FILE_UPLOAD
        assert error_info.message == "Test error message"
        assert error_info.user_message is not None
        assert error_info.technical_details is not None
        assert isinstance(error_info.timestamp, datetime)
        mock_st_error.assert_called_once()
    
    def test_handle_error_with_string(self):
        """Test handling of string error messages."""
        error_message = "File size exceeds limit"
        
        with patch('streamlit.error') as mock_st_error:
            error_info = self.error_handler.handle_error(
                error_message,
                ErrorCategory.FILE_UPLOAD,
                show_ui=True
            )
        
        assert error_info.category == ErrorCategory.FILE_UPLOAD
        assert error_info.message == error_message
        assert error_info.technical_details is None
        mock_st_error.assert_called_once()
    
    def test_severity_determination(self):
        """Test error severity determination logic."""
        # Test critical errors
        critical_error = "Out of memory error"
        severity = self.error_handler._determine_severity(critical_error, ErrorCategory.SYSTEM_ERROR)
        assert severity == ErrorSeverity.CRITICAL
        
        # Test high severity errors
        high_error = "API key authentication failed"
        severity = self.error_handler._determine_severity(high_error, ErrorCategory.API_ERROR)
        assert severity == ErrorSeverity.HIGH
        
        # Test medium severity errors
        medium_error = "Connection timeout"
        severity = self.error_handler._determine_severity(medium_error, ErrorCategory.QUERY_PROCESSING)
        assert severity == ErrorSeverity.MEDIUM
        
        # Test low severity errors
        low_error = "Chart rendering issue"
        severity = self.error_handler._determine_severity(low_error, ErrorCategory.VISUALIZATION)
        assert severity == ErrorSeverity.LOW
    
    def test_user_friendly_messages(self):
        """Test conversion of technical errors to user-friendly messages."""
        test_cases = [
            ("file size", "The file is too large. Please use a file smaller than 50MB."),
            ("unsupported format", "This file format is not supported. Please use CSV or Excel files."),
            ("api key", "API key issue. Please check your GEMINI_API_KEY configuration."),
            ("timeout", "The query is taking too long. Please try a simpler question."),
            ("matplotlib", "Error creating chart. Try asking for a different type of visualization.")
        ]
        
        for error_pattern, expected_message in test_cases:
            user_message = self.error_handler._get_user_friendly_message(
                error_pattern, ErrorCategory.FILE_UPLOAD
            )
            assert user_message == expected_message
    
    def test_recovery_suggestions(self):
        """Test recovery suggestion generation."""
        # Test memory error suggestions
        error_message = "Out of memory"
        suggestions = self.error_handler._get_recovery_suggestions(
            error_message, ErrorCategory.QUERY_PROCESSING, None
        )
        
        assert "Try using a smaller dataset" in suggestions
        assert "Close other applications to free up memory" in suggestions
        
        # Test API key error suggestions
        error_message = "API key invalid"
        suggestions = self.error_handler._get_recovery_suggestions(
            error_message, ErrorCategory.API_ERROR, None
        )
        
        assert "Check your API key configuration" in suggestions
        assert "Verify your API key is valid and active" in suggestions
    
    def test_file_upload_error_handling(self):
        """Test file upload specific error handling."""
        filename = "test_file.csv"
        error_message = "File size exceeds limit"
        
        with patch('streamlit.error') as mock_st_error:
            error_info = self.error_handler.handle_file_upload_error(
                error_message, filename, show_ui=True
            )
        
        assert error_info.category == ErrorCategory.FILE_UPLOAD
        assert error_info.message == error_message
        mock_st_error.assert_called_once()
    
    def test_query_processing_error_handling(self):
        """Test query processing specific error handling."""
        query = "Show me a chart"
        error_message = "Column not found"
        
        with patch('streamlit.error') as mock_st_error:
            error_info = self.error_handler.handle_query_processing_error(
                error_message, query, show_ui=True
            )
        
        assert error_info.category == ErrorCategory.QUERY_PROCESSING
        assert error_info.message == error_message
        mock_st_error.assert_called_once()
    
    def test_visualization_error_handling(self):
        """Test visualization specific error handling."""
        chart_type = "matplotlib"
        error_message = "Chart rendering failed"
        
        with patch('streamlit.error') as mock_st_error:
            error_info = self.error_handler.handle_visualization_error(
                error_message, chart_type, show_ui=True
            )
        
        assert error_info.category == ErrorCategory.VISUALIZATION
        assert error_info.message == error_message
        mock_st_error.assert_called_once()
    
    @patch('streamlit.error')
    @patch('streamlit.expander')
    def test_display_error_ui(self, mock_expander, mock_st_error):
        """Test error UI display functionality."""
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = Mock(return_value=mock_expander_context)
        mock_expander.return_value.__exit__ = Mock(return_value=None)
        
        error_info = ErrorInfo(
            category=ErrorCategory.FILE_UPLOAD,
            severity=ErrorSeverity.HIGH,
            message="Test error",
            user_message="User friendly error",
            suggestions=["Suggestion 1", "Suggestion 2"],
            timestamp=datetime.now()
        )
        
        self.error_handler._display_error_ui(error_info)
        
        mock_st_error.assert_called_once()
        mock_expander.assert_called_once()
    
    def test_no_ui_display(self):
        """Test that UI is not displayed when show_ui=False."""
        with patch('streamlit.error') as mock_st_error:
            self.error_handler.handle_error(
                "Test error",
                ErrorCategory.FILE_UPLOAD,
                show_ui=False
            )
        
        mock_st_error.assert_not_called()


class TestConvenienceFunctions:
    """Test cases for convenience functions."""
    
    def test_handle_file_error(self):
        """Test handle_file_error convenience function."""
        with patch('streamlit.error') as mock_st_error:
            error_info = handle_file_error("Test file error", "test.csv", show_ui=True)
        
        assert error_info.category == ErrorCategory.FILE_UPLOAD
        assert error_info.message == "Test file error"
        mock_st_error.assert_called_once()
    
    def test_handle_query_error(self):
        """Test handle_query_error convenience function."""
        with patch('streamlit.error') as mock_st_error:
            error_info = handle_query_error("Test query error", "test query", show_ui=True)
        
        assert error_info.category == ErrorCategory.QUERY_PROCESSING
        assert error_info.message == "Test query error"
        mock_st_error.assert_called_once()
    
    def test_handle_viz_error(self):
        """Test handle_viz_error convenience function."""
        with patch('streamlit.error') as mock_st_error:
            error_info = handle_viz_error("Test viz error", "matplotlib", show_ui=True)
        
        assert error_info.category == ErrorCategory.VISUALIZATION
        assert error_info.message == "Test viz error"
        mock_st_error.assert_called_once()


class TestSafeExecute:
    """Test cases for safe_execute function."""
    
    def test_safe_execute_success(self):
        """Test safe_execute with successful function execution."""
        def successful_function():
            return "success"
        
        result = safe_execute(
            successful_function,
            ErrorCategory.SYSTEM_ERROR,
            fallback_value="fallback"
        )
        
        assert result == "success"
    
    def test_safe_execute_failure(self):
        """Test safe_execute with failing function."""
        def failing_function():
            raise ValueError("Test error")
        
        with patch('streamlit.error') as mock_st_error:
            result = safe_execute(
                failing_function,
                ErrorCategory.SYSTEM_ERROR,
                fallback_value="fallback"
            )
        
        assert result == "fallback"
        mock_st_error.assert_called_once()
    
    def test_safe_execute_with_context(self):
        """Test safe_execute with context information."""
        def failing_function():
            raise ValueError("Test error")
        
        context = {"operation": "test_operation"}
        
        with patch('streamlit.error') as mock_st_error:
            result = safe_execute(
                failing_function,
                ErrorCategory.SYSTEM_ERROR,
                context=context,
                fallback_value="fallback"
            )
        
        assert result == "fallback"
        mock_st_error.assert_called_once()


class TestErrorPatterns:
    """Test cases for error pattern matching."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_file_error_patterns(self):
        """Test file-related error pattern matching."""
        test_cases = [
            ("file size exceeds limit", "file size"),
            ("unsupported file format", "unsupported format"),
            ("encoding error occurred", "encoding"),
            ("file appears corrupted", "corrupted"),
            ("empty file detected", "empty file")
        ]
        
        for error_message, expected_pattern in test_cases:
            user_message = self.error_handler._get_user_friendly_message(
                error_message, ErrorCategory.FILE_UPLOAD
            )
            # Check that the pattern was matched (not using fallback message)
            assert user_message != "There was a problem with your file. Please check the file format and try again."
    
    def test_query_error_patterns(self):
        """Test query-related error pattern matching."""
        test_cases = [
            ("api key invalid", "api key"),
            ("rate limit exceeded", "rate limit"),
            ("query timeout occurred", "timeout"),
            ("invalid query format", "invalid query"),
            ("no data available", "no data")
        ]
        
        for error_message, expected_pattern in test_cases:
            user_message = self.error_handler._get_user_friendly_message(
                error_message, ErrorCategory.QUERY_PROCESSING
            )
            # Check that the pattern was matched (not using fallback message)
            assert user_message != "Unable to process your query. Please try rephrasing your question."
    
    def test_visualization_error_patterns(self):
        """Test visualization-related error pattern matching."""
        test_cases = [
            ("matplotlib error occurred", "matplotlib"),
            ("plotly rendering failed", "plotly"),
            ("no suitable data for chart", "no suitable data"),
            ("too many data points", "too many points")
        ]
        
        for error_message, expected_pattern in test_cases:
            user_message = self.error_handler._get_user_friendly_message(
                error_message, ErrorCategory.VISUALIZATION
            )
            # Check that the pattern was matched (not using fallback message)
            assert user_message != "Could not create the visualization. The data might not be suitable for this chart type."


class TestErrorRecovery:
    """Test cases for error recovery functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    @patch('streamlit.button')
    @patch('streamlit.columns')
    @patch('streamlit.subheader')
    @patch('streamlit.markdown')
    def test_create_error_recovery_ui(self, mock_markdown, mock_subheader, mock_columns, mock_button):
        """Test error recovery UI creation."""
        mock_columns.return_value = [MagicMock(), MagicMock()]
        mock_button.return_value = False
        
        error_info = ErrorInfo(
            category=ErrorCategory.FILE_UPLOAD,
            severity=ErrorSeverity.MEDIUM,
            message="Test error",
            user_message="User friendly error",
            suggestions=["Suggestion 1", "Suggestion 2"],
            timestamp=datetime.now()
        )
        
        self.error_handler.create_error_recovery_ui(error_info)
        
        mock_subheader.assert_called_once()
        mock_columns.assert_called_once()
        assert mock_button.call_count == 2  # Try Again and Clear Session buttons
        assert mock_markdown.call_count >= 3  # Error details and suggestions


if __name__ == "__main__":
    pytest.main([__file__])