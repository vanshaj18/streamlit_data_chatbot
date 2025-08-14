"""
Unit tests for plotting-specific error handling functionality.
"""

import pytest
from datetime import datetime
from unittest.mock import patch, MagicMock

from utils.error_handler import (
    ErrorHandler, PlottingErrorType, PlottingErrorInfo, ErrorCategory, ErrorSeverity,
    handle_plotting_error
)


class TestPlottingErrorClassification:
    """Test plotting error classification methods."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_classify_matplotlib_errors(self):
        """Test classification of matplotlib-specific errors."""
        matplotlib_errors = [
            "matplotlib.pyplot error occurred",
            "Figure creation failed",
            "No display name and no $DISPLAY environment variable",
            "Backend TkAgg is not available",
            "Error in savefig operation"
        ]
        
        for error in matplotlib_errors:
            error_type = self.error_handler._classify_plotting_error(error)
            assert error_type == PlottingErrorType.MATPLOTLIB_ERROR
    
    def test_classify_pandasai_errors(self):
        """Test classification of PandasAI plotting errors."""
        pandasai_errors = [
            "PandasAI SmartDataframe error",
            "Failed to generate_code for plotting",
            "Base64 decode error in chart",
            "SQL table name resolution failed",
            "PandasAI chat response invalid"
        ]
        
        for error in pandasai_errors:
            error_type = self.error_handler._classify_plotting_error(error)
            assert error_type == PlottingErrorType.PANDASAI_PLOTTING
    
    def test_classify_data_compatibility_errors(self):
        """Test classification of data compatibility errors."""
        data_errors = [
            "Column 'sales' not found in dataframe",
            "Missing column for chart generation",
            "Data type incompatible with chart",
            "Insufficient data points for visualization",
            "Empty dataframe provided",
            "No numeric data available",
            "String cannot be converted to numeric"
        ]
        
        for error in data_errors:
            error_type = self.error_handler._classify_plotting_error(error)
            assert error_type == PlottingErrorType.DATA_COMPATIBILITY
    
    def test_classify_memory_errors(self):
        """Test classification of memory exhaustion errors."""
        memory_errors = [
            "Out of memory during chart creation",
            "MemoryError: allocation failed",
            "Dataset too large for visualization",
            "Memory allocation failed"
        ]
        
        for error in memory_errors:
            error_type = self.error_handler._classify_plotting_error(error)
            assert error_type == PlottingErrorType.MEMORY_EXHAUSTION
    
    def test_classify_timeout_errors(self):
        """Test classification of timeout errors."""
        timeout_errors = [
            "Chart generation timed out",
            "Operation timed out after 30 seconds",
            "Time limit exceeded for plotting",
            "Chart took too long to generate"
        ]
        
        for error in timeout_errors:
            error_type = self.error_handler._classify_plotting_error(error)
            assert error_type == PlottingErrorType.TIMEOUT
    
    def test_classify_invalid_chart_type_errors(self):
        """Test classification of invalid chart type errors."""
        chart_type_errors = [
            "Invalid chart type specified",
            "Unsupported chart type: bubble3d",
            "Unknown plot type requested",
            "Chart type not recognized"
        ]
        
        for error in chart_type_errors:
            error_type = self.error_handler._classify_plotting_error(error)
            assert error_type == PlottingErrorType.INVALID_CHART_TYPE
    
    def test_classify_general_chart_generation_errors(self):
        """Test classification of general chart generation errors."""
        general_errors = [
            "Generic plotting error",
            "Chart creation failed",
            "Visualization error occurred"
        ]
        
        for error in general_errors:
            error_type = self.error_handler._classify_plotting_error(error)
            assert error_type == PlottingErrorType.CHART_GENERATION


class TestPlottingSeverityDetermination:
    """Test plotting error severity determination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_memory_errors_are_critical(self):
        """Test that memory errors are classified as critical."""
        severity = self.error_handler._determine_plotting_severity(
            "Out of memory", PlottingErrorType.MEMORY_EXHAUSTION
        )
        assert severity == ErrorSeverity.CRITICAL
    
    def test_pandasai_errors_are_high(self):
        """Test that PandasAI errors are classified as high severity."""
        severity = self.error_handler._determine_plotting_severity(
            "PandasAI error", PlottingErrorType.PANDASAI_PLOTTING
        )
        assert severity == ErrorSeverity.HIGH
    
    def test_matplotlib_errors_are_high(self):
        """Test that matplotlib errors are classified as high severity."""
        severity = self.error_handler._determine_plotting_severity(
            "Matplotlib error", PlottingErrorType.MATPLOTLIB_ERROR
        )
        assert severity == ErrorSeverity.HIGH
    
    def test_data_compatibility_errors_are_medium(self):
        """Test that data compatibility errors are classified as medium severity."""
        severity = self.error_handler._determine_plotting_severity(
            "Column not found", PlottingErrorType.DATA_COMPATIBILITY
        )
        assert severity == ErrorSeverity.MEDIUM
    
    def test_timeout_errors_are_medium(self):
        """Test that timeout errors are classified as medium severity."""
        severity = self.error_handler._determine_plotting_severity(
            "Timeout", PlottingErrorType.TIMEOUT
        )
        assert severity == ErrorSeverity.MEDIUM
    
    def test_other_errors_are_low(self):
        """Test that other plotting errors are classified as low severity."""
        severity = self.error_handler._determine_plotting_severity(
            "General error", PlottingErrorType.CHART_GENERATION
        )
        assert severity == ErrorSeverity.LOW


class TestPlottingUserMessages:
    """Test plotting error user message generation."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_matplotlib_error_message(self):
        """Test matplotlib error user message."""
        message = self.error_handler._get_plotting_user_message(
            "Backend error", PlottingErrorType.MATPLOTLIB_ERROR, "bar"
        )
        assert "chart rendering engine" in message
        assert "bar chart" in message
        assert "display or graphics issues" in message
    
    def test_pandasai_error_message(self):
        """Test PandasAI error user message."""
        message = self.error_handler._get_plotting_user_message(
            "Code generation failed", PlottingErrorType.PANDASAI_PLOTTING, "pie"
        )
        assert "AI had trouble generating" in message
        assert "pie chart" in message
        assert "plotting code" in message
    
    def test_data_compatibility_error_message(self):
        """Test data compatibility error user message."""
        message = self.error_handler._get_plotting_user_message(
            "Column missing", PlottingErrorType.DATA_COMPATIBILITY, "scatter"
        )
        assert "data isn't compatible" in message
        assert "scatter chart" in message
        assert "missing required columns" in message
    
    def test_memory_error_message(self):
        """Test memory error user message."""
        message = self.error_handler._get_plotting_user_message(
            "Out of memory", PlottingErrorType.MEMORY_EXHAUSTION, "histogram"
        )
        assert "Not enough memory" in message
        assert "histogram chart" in message
        assert "too large" in message
    
    def test_timeout_error_message(self):
        """Test timeout error user message."""
        message = self.error_handler._get_plotting_user_message(
            "Timeout", PlottingErrorType.TIMEOUT, "line"
        )
        assert "taking too long" in message
        assert "line chart" in message
        assert "too complex or large" in message
    
    def test_invalid_chart_type_message(self):
        """Test invalid chart type error message."""
        message = self.error_handler._get_plotting_user_message(
            "Invalid chart", PlottingErrorType.INVALID_CHART_TYPE, "bubble3d"
        )
        assert "not supported" in message
        assert "bubble3d" in message
    
    def test_general_error_message(self):
        """Test general chart generation error message."""
        message = self.error_handler._get_plotting_user_message(
            "General error", PlottingErrorType.CHART_GENERATION
        )
        assert "Unable to generate" in message
        assert "data or chart configuration" in message


class TestPlottingRecoverySuggestions:
    """Test plotting error recovery suggestions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_matplotlib_error_suggestions(self):
        """Test matplotlib error recovery suggestions."""
        suggestions = self.error_handler._get_plotting_recovery_suggestions(
            "Backend error", PlottingErrorType.MATPLOTLIB_ERROR
        )
        assert any("refresh" in s.lower() for s in suggestions)
        assert any("simpler chart" in s.lower() for s in suggestions)
        assert any("browser" in s.lower() for s in suggestions)
    
    def test_pandasai_error_suggestions(self):
        """Test PandasAI error recovery suggestions."""
        suggestions = self.error_handler._get_plotting_recovery_suggestions(
            "Code generation failed", PlottingErrorType.PANDASAI_PLOTTING
        )
        assert any("rephras" in s.lower() for s in suggestions)
        assert any("bar chart" in s.lower() for s in suggestions)
        assert any("columns" in s.lower() for s in suggestions)
    
    def test_data_compatibility_suggestions(self):
        """Test data compatibility error recovery suggestions."""
        suggestions = self.error_handler._get_plotting_recovery_suggestions(
            "Column not found", PlottingErrorType.DATA_COMPATIBILITY,
            data_columns=["name", "age", "salary"]
        )
        assert any("columns" in s.lower() for s in suggestions)
        assert any("different chart type" in s.lower() for s in suggestions)
        assert any("name, age, salary" in s for s in suggestions)
    
    def test_memory_error_suggestions(self):
        """Test memory error recovery suggestions."""
        suggestions = self.error_handler._get_plotting_recovery_suggestions(
            "Out of memory", PlottingErrorType.MEMORY_EXHAUSTION
        )
        assert any("filter" in s.lower() for s in suggestions)
        assert any("sample" in s.lower() for s in suggestions)
        assert any("aggregat" in s.lower() for s in suggestions)
        assert any("browser tabs" in s.lower() for s in suggestions)
    
    def test_timeout_error_suggestions(self):
        """Test timeout error recovery suggestions."""
        suggestions = self.error_handler._get_plotting_recovery_suggestions(
            "Timeout", PlottingErrorType.TIMEOUT
        )
        assert any("simpler" in s.lower() for s in suggestions)
        assert any("filter" in s.lower() for s in suggestions)
        assert any("connection" in s.lower() for s in suggestions)
    
    def test_invalid_chart_type_suggestions(self):
        """Test invalid chart type error recovery suggestions."""
        suggestions = self.error_handler._get_plotting_recovery_suggestions(
            "Invalid chart", PlottingErrorType.INVALID_CHART_TYPE
        )
        assert any("bar chart" in s.lower() for s in suggestions)
        assert any("simpler chart names" in s.lower() for s in suggestions)
        assert any("what charts" in s.lower() for s in suggestions)
    
    def test_long_query_suggestion(self):
        """Test suggestion for long queries."""
        long_query = "a" * 150  # Long query
        suggestions = self.error_handler._get_plotting_recovery_suggestions(
            "Error", PlottingErrorType.CHART_GENERATION, query=long_query
        )
        assert any("shorter" in s.lower() for s in suggestions)


class TestFallbackAvailability:
    """Test fallback availability determination."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_memory_errors_no_fallback(self):
        """Test that memory errors don't have fallback available."""
        available = self.error_handler._is_fallback_available(
            PlottingErrorType.MEMORY_EXHAUSTION
        )
        assert not available
    
    def test_common_chart_types_have_fallback(self):
        """Test that common chart types have fallback available."""
        common_types = ['bar', 'line', 'scatter', 'histogram', 'pie']
        for chart_type in common_types:
            available = self.error_handler._is_fallback_available(
                PlottingErrorType.PANDASAI_PLOTTING, chart_type
            )
            assert available
    
    def test_pandasai_errors_have_fallback(self):
        """Test that PandasAI errors have fallback available."""
        available = self.error_handler._is_fallback_available(
            PlottingErrorType.PANDASAI_PLOTTING
        )
        assert available
    
    def test_matplotlib_errors_have_fallback(self):
        """Test that matplotlib errors have fallback available."""
        available = self.error_handler._is_fallback_available(
            PlottingErrorType.MATPLOTLIB_ERROR
        )
        assert available


class TestAlternativeChartSuggestions:
    """Test alternative chart type suggestions."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_pie_chart_alternative(self):
        """Test alternative suggestion for pie chart data compatibility issues."""
        suggestion = self.error_handler._suggest_alternative_chart_type(
            PlottingErrorType.DATA_COMPATIBILITY, "pie"
        )
        assert suggestion == "bar chart"
    
    def test_scatter_chart_alternative(self):
        """Test alternative suggestion for scatter chart data compatibility issues."""
        suggestion = self.error_handler._suggest_alternative_chart_type(
            PlottingErrorType.DATA_COMPATIBILITY, "scatter"
        )
        assert suggestion == "line chart"
    
    def test_memory_error_alternative(self):
        """Test alternative suggestion for memory errors."""
        suggestion = self.error_handler._suggest_alternative_chart_type(
            PlottingErrorType.MEMORY_EXHAUSTION
        )
        assert "aggregated" in suggestion
    
    def test_invalid_chart_type_alternative(self):
        """Test alternative suggestion for invalid chart types."""
        suggestion = self.error_handler._suggest_alternative_chart_type(
            PlottingErrorType.INVALID_CHART_TYPE
        )
        assert suggestion == "bar chart"


class TestPlottingErrorHandling:
    """Test complete plotting error handling workflow."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    @patch('streamlit.warning')
    @patch('streamlit.expander')
    def test_handle_plotting_error_complete_workflow(self, mock_expander, mock_st_warning):
        """Test complete plotting error handling workflow."""
        # Mock streamlit components
        mock_expander_context = MagicMock()
        mock_expander.return_value.__enter__ = MagicMock(return_value=mock_expander_context)
        mock_expander.return_value.__exit__ = MagicMock(return_value=None)
        
        error_info = self.error_handler.handle_plotting_error(
            "Column 'sales' not found",
            chart_type="bar",
            data_columns=["name", "age", "salary"],
            query="show me sales data",
            show_ui=True
        )
        
        # Verify error info structure
        assert isinstance(error_info, PlottingErrorInfo)
        assert error_info.category == ErrorCategory.PLOTTING
        assert error_info.plotting_error_type == PlottingErrorType.DATA_COMPATIBILITY
        assert error_info.chart_type == "bar"
        assert error_info.data_columns == ["name", "age", "salary"]
        assert error_info.fallback_available
        assert error_info.suggested_chart_type == "bar chart"
        
        # Verify user message
        assert "data isn't compatible" in error_info.user_message
        assert "bar chart" in error_info.user_message
        
        # Verify suggestions
        assert len(error_info.suggestions) > 0
        assert any("columns" in s.lower() for s in error_info.suggestions)
        
        # Verify UI was called (data compatibility errors are MEDIUM severity -> st.warning)
        mock_st_warning.assert_called_once()
        mock_expander.assert_called_once()
    
    def test_handle_plotting_error_with_exception(self):
        """Test plotting error handling with Exception object."""
        try:
            raise ValueError("Matplotlib backend error")
        except ValueError as e:
            error_info = self.error_handler.handle_plotting_error(
                e, chart_type="line", show_ui=False
            )
        
        assert error_info.plotting_error_type == PlottingErrorType.MATPLOTLIB_ERROR
        assert error_info.technical_details is not None
        assert "Traceback" in error_info.technical_details
    
    def test_convenience_function(self):
        """Test the convenience function for plotting errors."""
        error_info = handle_plotting_error(
            "PandasAI generation failed",
            chart_type="scatter",
            show_ui=False
        )
        
        assert isinstance(error_info, PlottingErrorInfo)
        assert error_info.plotting_error_type == PlottingErrorType.PANDASAI_PLOTTING
        assert error_info.chart_type == "scatter"


class TestPlottingErrorIntegration:
    """Test integration with existing error handling system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.error_handler = ErrorHandler()
    
    def test_plotting_category_in_error_patterns(self):
        """Test that plotting category is handled in error patterns."""
        # Test that plotting errors get appropriate user messages
        user_message = self.error_handler._get_user_friendly_message(
            "Unknown plotting error", ErrorCategory.PLOTTING
        )
        assert "problem generating the chart" in user_message
    
    def test_plotting_category_in_recovery_suggestions(self):
        """Test that plotting category has recovery suggestions."""
        suggestions = self.error_handler._recovery_suggestions.get(ErrorCategory.PLOTTING)
        assert suggestions is not None
        assert len(suggestions) > 0
        assert any("basic chart type" in s.lower() for s in suggestions)
    
    def test_plotting_severity_determination(self):
        """Test that plotting category gets appropriate severity."""
        severity = self.error_handler._determine_severity(
            "General plotting error", ErrorCategory.PLOTTING
        )
        assert severity == ErrorSeverity.MEDIUM


if __name__ == "__main__":
    pytest.main([__file__])