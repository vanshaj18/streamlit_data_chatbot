"""
Unit tests for visualization rendering system.
"""

import pytest
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.figure
import plotly.graph_objects as go
import plotly.express as px
from unittest.mock import Mock, patch, MagicMock
import streamlit as st
import io
import base64

from components.visualization import (
    render_matplotlib_chart,
    render_plotly_chart,
    render_chart,
    handle_chart_error,
    create_sample_charts,
    validate_chart_data,
    _get_user_friendly_error_message,
    _render_image_data,
    _render_fallback_content
)


class TestMatplotlibRendering:
    """Test matplotlib chart rendering functionality."""
    
    def test_render_matplotlib_chart_success(self):
        """Test successful matplotlib chart rendering."""
        # Create a simple matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        ax.set_title("Test Chart")
        
        with patch('streamlit.pyplot') as mock_pyplot:
            result = render_matplotlib_chart(fig)
            
            assert result is True
            mock_pyplot.assert_called_once()
    
    def test_render_matplotlib_chart_with_title(self):
        """Test matplotlib chart rendering with title."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with patch('streamlit.pyplot') as mock_pyplot, \
             patch('streamlit.subheader') as mock_subheader:
            
            result = render_matplotlib_chart(fig, title="Custom Title")
            
            assert result is True
            mock_subheader.assert_called_once_with("Custom Title")
            mock_pyplot.assert_called_once()
    
    def test_render_matplotlib_chart_none_figure(self):
        """Test matplotlib chart rendering with None figure."""
        result = render_matplotlib_chart(None)
        assert result is False
    
    def test_render_matplotlib_chart_invalid_type(self):
        """Test matplotlib chart rendering with invalid type."""
        result = render_matplotlib_chart("not a figure")
        assert result is False
    
    def test_render_matplotlib_chart_exception(self):
        """Test matplotlib chart rendering with exception."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with patch('streamlit.pyplot', side_effect=Exception("Streamlit error")):
            result = render_matplotlib_chart(fig)
            assert result is False


class TestPlotlyRendering:
    """Test plotly chart rendering functionality."""
    
    def test_render_plotly_chart_success(self):
        """Test successful plotly chart rendering."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        with patch('streamlit.plotly_chart') as mock_plotly:
            result = render_plotly_chart(fig)
            
            assert result is True
            mock_plotly.assert_called_once()
    
    def test_render_plotly_chart_with_title(self):
        """Test plotly chart rendering with title."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        with patch('streamlit.plotly_chart') as mock_plotly, \
             patch('streamlit.subheader') as mock_subheader:
            
            result = render_plotly_chart(fig, title="Custom Title", height=400)
            
            assert result is True
            mock_subheader.assert_called_once_with("Custom Title")
            mock_plotly.assert_called_once()
    
    def test_render_plotly_chart_none_figure(self):
        """Test plotly chart rendering with None figure."""
        result = render_plotly_chart(None)
        assert result is False
    
    def test_render_plotly_chart_invalid_type(self):
        """Test plotly chart rendering with invalid type."""
        result = render_plotly_chart("not a figure")
        assert result is False
    
    def test_render_plotly_chart_exception(self):
        """Test plotly chart rendering with exception."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        with patch('streamlit.plotly_chart', side_effect=Exception("Streamlit error")):
            result = render_plotly_chart(fig)
            assert result is False


class TestUniversalChartRenderer:
    """Test the universal chart rendering system."""
    
    def test_render_chart_matplotlib(self):
        """Test universal renderer with matplotlib figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with patch('components.visualization.render_matplotlib_chart', return_value=True) as mock_render:
            result = render_chart(fig, title="Test Chart")
            
            assert result is True
            mock_render.assert_called_once_with(fig, "Test Chart")
    
    def test_render_chart_plotly(self):
        """Test universal renderer with plotly figure."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        with patch('components.visualization.render_plotly_chart', return_value=True) as mock_render:
            result = render_chart(fig, title="Test Chart")
            
            assert result is True
            mock_render.assert_called_once_with(fig, "Test Chart")
    
    def test_render_chart_dataframe(self):
        """Test universal renderer with DataFrame."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
        
        with patch('streamlit.subheader') as mock_subheader, \
             patch('streamlit.dataframe') as mock_dataframe:
            
            result = render_chart(df, title="Data Table")
            
            assert result is True
            mock_subheader.assert_called_once_with("Data Table")
            mock_dataframe.assert_called_once()
    
    def test_render_chart_with_show_method(self):
        """Test universal renderer with object having show method."""
        mock_chart = Mock()
        mock_chart.show = Mock()
        
        with patch('streamlit.plotly_chart') as mock_plotly:
            result = render_chart(mock_chart)
            
            assert result is True
            mock_plotly.assert_called_once()
    
    def test_render_chart_with_savefig_method(self):
        """Test universal renderer with object having savefig method."""
        # Create a mock that only has savefig method, not show
        mock_chart = Mock(spec=['savefig'])
        mock_chart.savefig = Mock()
        
        # Patch the render_chart function to test the savefig path specifically
        with patch('components.visualization.st.pyplot') as mock_pyplot, \
             patch('components.visualization.st.subheader') as mock_subheader:
            
            # Mock pyplot to succeed
            mock_pyplot.return_value = None
            
            result = render_chart(mock_chart, title="Test Chart")
            
            # Should succeed with mocked streamlit functions
            assert result is True
            mock_pyplot.assert_called_once_with(mock_chart)
            mock_subheader.assert_called_once_with("Test Chart")
    
    def test_render_chart_unknown_type(self):
        """Test universal renderer with unknown chart type."""
        unknown_data = {"some": "data"}
        
        with patch('components.visualization._render_fallback_content', return_value=True) as mock_fallback:
            result = render_chart(unknown_data)
            
            assert result is True
            mock_fallback.assert_called_once()
    
    def test_render_chart_exception(self):
        """Test universal renderer with exception."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with patch('components.visualization.render_matplotlib_chart', side_effect=Exception("Error")), \
             patch('components.visualization.handle_chart_error', return_value=False) as mock_error:
            
            result = render_chart(fig)
            
            assert result is False
            mock_error.assert_called_once()


class TestImageRendering:
    """Test image data rendering functionality."""
    
    def test_render_image_data_base64(self):
        """Test rendering base64 encoded image."""
        base64_data = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8/5+hHgAHggJ/PchI7wAAAABJRU5ErkJggg=="
        
        with patch('streamlit.markdown') as mock_markdown:
            result = _render_image_data(base64_data, title="Test Image")
            
            assert result is True
            mock_markdown.assert_called_once()
    
    def test_render_image_data_bytes(self):
        """Test rendering image bytes."""
        image_bytes = b"fake image data"
        
        with patch('streamlit.image') as mock_image:
            result = _render_image_data(image_bytes)
            
            assert result is True
            mock_image.assert_called_once()
    
    def test_render_image_data_exception(self):
        """Test image rendering with exception."""
        with patch('streamlit.image', side_effect=Exception("Image error")):
            result = _render_image_data(b"fake data")
            assert result is False


class TestFallbackRendering:
    """Test fallback content rendering."""
    
    def test_render_fallback_content_dict(self):
        """Test fallback rendering with dictionary."""
        content = {"key": "value", "number": 42}
        
        with patch('streamlit.json') as mock_json, \
             patch('streamlit.info') as mock_info:
            
            result = _render_fallback_content(content)
            
            assert result is True
            mock_json.assert_called_once_with(content)
            mock_info.assert_called_once()
    
    def test_render_fallback_content_object_with_dict(self):
        """Test fallback rendering with object having __dict__."""
        class TestObject:
            def __init__(self):
                self.attr1 = "value1"
                self.attr2 = 42
        
        obj = TestObject()
        
        with patch('streamlit.json') as mock_json:
            result = _render_fallback_content(obj)
            
            assert result is True
            mock_json.assert_called_once()
    
    def test_render_fallback_content_string(self):
        """Test fallback rendering with string."""
        content = "Simple text content"
        
        with patch('streamlit.text') as mock_text:
            result = _render_fallback_content(content)
            
            assert result is True
            mock_text.assert_called_once_with(content)
    
    def test_render_fallback_content_custom_message(self):
        """Test fallback rendering with custom message."""
        content = "test"
        custom_msg = "Custom fallback message"
        
        with patch('streamlit.info') as mock_info:
            result = _render_fallback_content(content, fallback_message=custom_msg)
            
            assert result is True
            mock_info.assert_called_once_with(custom_msg)


class TestErrorHandling:
    """Test error handling functionality."""
    
    def test_handle_chart_error_basic(self):
        """Test basic error handling."""
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.expander') as mock_expander:
            
            mock_expander.return_value.__enter__ = Mock()
            mock_expander.return_value.__exit__ = Mock()
            
            result = handle_chart_error("Test error message")
            
            assert result is False
            mock_error.assert_called_once()
    
    def test_handle_chart_error_custom_message(self):
        """Test error handling with custom message."""
        with patch('streamlit.error') as mock_error, \
             patch('streamlit.expander'):
            
            result = handle_chart_error("Technical error", "User friendly message")
            
            assert result is False
            # Check that the error message contains the user-friendly message
            error_call_args = mock_error.call_args[0][0]
            assert "User friendly message" in error_call_args
    
    def test_get_user_friendly_error_message_memory(self):
        """Test user-friendly error message for memory errors."""
        error_msg = "Out of memory error occurred"
        result = _get_user_friendly_error_message(error_msg)
        assert "memory" in result.lower()
    
    def test_get_user_friendly_error_message_column(self):
        """Test user-friendly error message for column errors."""
        error_msg = "Column 'missing_col' not found in DataFrame"
        result = _get_user_friendly_error_message(error_msg)
        assert "column" in result.lower()
    
    def test_get_user_friendly_error_message_custom(self):
        """Test user-friendly error message with custom message."""
        result = _get_user_friendly_error_message("Any error", "Custom message")
        assert result == "Custom message"


class TestChartValidation:
    """Test chart data validation."""
    
    def test_validate_chart_data_matplotlib(self):
        """Test validation of matplotlib figure."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        is_valid, error_msg = validate_chart_data(fig)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_chart_data_matplotlib_no_axes(self):
        """Test validation of matplotlib figure with no axes."""
        fig = plt.figure()  # Empty figure with no axes
        
        is_valid, error_msg = validate_chart_data(fig)
        
        assert is_valid is False
        assert "no axes" in error_msg.lower()
    
    def test_validate_chart_data_plotly(self):
        """Test validation of plotly figure."""
        fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        is_valid, error_msg = validate_chart_data(fig)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_chart_data_plotly_no_data(self):
        """Test validation of plotly figure with no data."""
        fig = go.Figure()  # Empty figure
        
        is_valid, error_msg = validate_chart_data(fig)
        
        assert is_valid is False
        assert "no data" in error_msg.lower()
    
    def test_validate_chart_data_dataframe(self):
        """Test validation of DataFrame."""
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
        
        is_valid, error_msg = validate_chart_data(df)
        
        assert is_valid is True
        assert error_msg == ""
    
    def test_validate_chart_data_empty_dataframe(self):
        """Test validation of empty DataFrame."""
        df = pd.DataFrame()
        
        is_valid, error_msg = validate_chart_data(df)
        
        assert is_valid is False
        assert "empty" in error_msg.lower()
    
    def test_validate_chart_data_none(self):
        """Test validation of None data."""
        is_valid, error_msg = validate_chart_data(None)
        
        assert is_valid is False
        assert "none" in error_msg.lower()
    
    def test_validate_chart_data_other_type(self):
        """Test validation of other data types."""
        is_valid, error_msg = validate_chart_data("some string")
        
        assert is_valid is True  # Other types are assumed valid if not None
        assert error_msg == ""


class TestSampleCharts:
    """Test sample chart creation."""
    
    def test_create_sample_charts(self):
        """Test creation of sample charts."""
        charts = create_sample_charts()
        
        assert 'matplotlib' in charts
        assert 'plotly' in charts
        assert 'dataframe' in charts
        
        # Validate matplotlib chart
        assert isinstance(charts['matplotlib'], matplotlib.figure.Figure)
        
        # Validate plotly chart
        assert isinstance(charts['plotly'], go.Figure)
        
        # Validate dataframe
        assert isinstance(charts['dataframe'], pd.DataFrame)
        assert not charts['dataframe'].empty


class TestIntegrationScenarios:
    """Test integration scenarios with real data."""
    
    def test_render_chart_integration_matplotlib(self):
        """Integration test with real matplotlib chart."""
        # Create sample data
        df = pd.DataFrame({
            'x': range(5),
            'y': [i**2 for i in range(5)]
        })
        
        # Create matplotlib chart
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df['x'], df['y'], marker='o')
        ax.set_title('Integration Test Chart')
        
        with patch('streamlit.pyplot') as mock_pyplot:
            result = render_chart(fig, title="Integration Test")
            
            assert result is True
            mock_pyplot.assert_called_once()
    
    def test_render_chart_integration_plotly(self):
        """Integration test with real plotly chart."""
        # Create sample data
        df = pd.DataFrame({
            'x': range(5),
            'y': [i**2 for i in range(5)],
            'category': ['A', 'B', 'A', 'B', 'A']
        })
        
        # Create plotly chart
        fig = px.scatter(df, x='x', y='y', color='category', title='Integration Test')
        
        with patch('streamlit.plotly_chart') as mock_plotly:
            result = render_chart(fig, title="Integration Test")
            
            assert result is True
            mock_plotly.assert_called_once()
    
    def test_error_recovery_scenario(self):
        """Test error recovery in realistic scenario."""
        # Simulate a chart that fails to render
        problematic_data = Mock()
        problematic_data.savefig = Mock(side_effect=Exception("Rendering failed"))
        
        with patch('streamlit.pyplot', side_effect=Exception("Streamlit error")), \
             patch('streamlit.error') as mock_error:
            
            result = render_chart(problematic_data)
            
            assert result is False
            mock_error.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])


class TestChatInterfaceIntegration:
    """Test integration between visualization system and chat interface."""
    
    def test_chat_interface_chart_rendering(self):
        """Test that chat interface properly uses visualization system."""
        from components.chat_interface import _render_chart
        
        # Create a matplotlib figure
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with patch('components.visualization.render_chart', return_value=True) as mock_render:
            _render_chart(fig)
            
            # Should call the visualization system's render_chart
            mock_render.assert_called_once_with(
                chart_data=fig,
                title=None,
                fallback_message="The chart could not be displayed in the chat."
            )
    
    def test_chat_interface_chart_error_handling(self):
        """Test that chat interface handles chart errors properly."""
        from components.chat_interface import _render_chart
        
        # Create problematic chart data
        problematic_data = "not a chart"
        
        with patch('components.visualization.render_chart', return_value=False) as mock_render:
            _render_chart(problematic_data)
            
            # Should attempt to render and handle failure gracefully
            mock_render.assert_called_once()
    
    def test_chat_interface_chart_exception_handling(self):
        """Test that chat interface handles exceptions in chart rendering."""
        from components.chat_interface import _render_chart
        
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with patch('components.visualization.render_chart', side_effect=Exception("Render error")), \
             patch('components.visualization.handle_chart_error', return_value=False) as mock_error:
            
            _render_chart(fig)
            
            # Should handle the exception and call error handler
            mock_error.assert_called_once()


class TestVisualizationSystemIntegration:
    """Test integration scenarios with the complete visualization system."""
    
    def test_end_to_end_matplotlib_rendering(self):
        """Test complete matplotlib rendering pipeline."""
        # Create sample data and chart
        df = pd.DataFrame({
            'x': range(10),
            'y': [i**2 for i in range(10)]
        })
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(df['x'], df['y'], marker='o', linewidth=2)
        ax.set_title('Sample Quadratic Function')
        ax.set_xlabel('X Values')
        ax.set_ylabel('Y Values')
        ax.grid(True, alpha=0.3)
        
        # Test validation
        is_valid, error_msg = validate_chart_data(fig)
        assert is_valid is True
        assert error_msg == ""
        
        # Test rendering
        with patch('streamlit.pyplot') as mock_pyplot:
            result = render_chart(fig, title="Integration Test Chart")
            
            assert result is True
            mock_pyplot.assert_called_once()
    
    def test_end_to_end_plotly_rendering(self):
        """Test complete plotly rendering pipeline."""
        # Create sample data and chart
        df = pd.DataFrame({
            'x': range(10),
            'y': [i**2 for i in range(10)],
            'category': ['A', 'B'] * 5
        })
        
        fig = px.scatter(
            df, 
            x='x', 
            y='y', 
            color='category',
            title='Sample Scatter Plot',
            hover_data=['x', 'y']
        )
        
        # Test validation
        is_valid, error_msg = validate_chart_data(fig)
        assert is_valid is True
        assert error_msg == ""
        
        # Test rendering
        with patch('streamlit.plotly_chart') as mock_plotly:
            result = render_chart(fig, title="Integration Test Chart")
            
            assert result is True
            mock_plotly.assert_called_once()
    
    def test_dataframe_table_rendering(self):
        """Test DataFrame rendering as table."""
        # Create sample DataFrame
        df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        })
        
        # Test validation
        is_valid, error_msg = validate_chart_data(df)
        assert is_valid is True
        assert error_msg == ""
        
        # Test rendering
        with patch('streamlit.dataframe') as mock_dataframe, \
             patch('streamlit.subheader') as mock_subheader:
            
            result = render_chart(df, title="Sample Data Table")
            
            assert result is True
            mock_subheader.assert_called_once_with("Sample Data Table")
            mock_dataframe.assert_called_once_with(df, use_container_width=True)
    
    def test_error_recovery_with_fallback(self):
        """Test error recovery and fallback mechanisms."""
        # Create a chart that will fail to render
        problematic_chart = Mock()
        problematic_chart.savefig = Mock(side_effect=Exception("Rendering failed"))
        
        # Remove show method to force savefig path
        del problematic_chart.show
        
        with patch('streamlit.pyplot', side_effect=Exception("Streamlit error")), \
             patch('components.visualization._render_fallback_content', return_value=True) as mock_fallback:
            
            result = render_chart(problematic_chart, fallback_message="Custom fallback")
            
            # Should fall back to fallback content
            assert result is True
            mock_fallback.assert_called_once()
    
    def test_comprehensive_error_handling(self):
        """Test comprehensive error handling across the system."""
        test_cases = [
            ("memory error", "memory"),
            ("column 'missing' not found", "column"),
            ("matplotlib error occurred", "matplotlib"),
            ("plotly rendering failed", "interactive chart"),  # plotly errors mention "interactive chart"
            ("type error", "data type mismatch"),  # type errors mention data type mismatch
            ("unknown rendering issue", "unable to create")  # generic fallback
        ]
        
        for error_input, expected_keyword in test_cases:
            user_friendly = _get_user_friendly_error_message(error_input)
            assert expected_keyword.lower() in user_friendly.lower()
    
    def test_sample_charts_integration(self):
        """Test that sample charts work with the rendering system."""
        sample_charts = create_sample_charts()
        
        # Test matplotlib sample
        with patch('streamlit.pyplot') as mock_pyplot:
            result = render_chart(sample_charts['matplotlib'])
            assert result is True
            mock_pyplot.assert_called_once()
        
        # Test plotly sample
        with patch('streamlit.plotly_chart') as mock_plotly:
            result = render_chart(sample_charts['plotly'])
            assert result is True
            mock_plotly.assert_called_once()
        
        # Test dataframe sample
        with patch('streamlit.dataframe') as mock_dataframe:
            result = render_chart(sample_charts['dataframe'])
            assert result is True
            mock_dataframe.assert_called_once()


class TestVisualizationPerformance:
    """Test performance aspects of the visualization system."""
    
    def test_large_dataframe_handling(self):
        """Test handling of large DataFrames."""
        # Create a large DataFrame
        large_df = pd.DataFrame({
            'x': range(1000),
            'y': [i**2 for i in range(1000)],
            'category': ['A', 'B', 'C', 'D'] * 250
        })
        
        # Test validation
        is_valid, error_msg = validate_chart_data(large_df)
        assert is_valid is True
        
        # Test rendering (should handle large data gracefully)
        with patch('streamlit.dataframe') as mock_dataframe:
            result = render_chart(large_df)
            assert result is True
            mock_dataframe.assert_called_once()
    
    def test_memory_cleanup(self):
        """Test that matplotlib figures are properly cleaned up."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 4, 9])
        
        with patch('streamlit.pyplot') as mock_pyplot, \
             patch('matplotlib.pyplot.close') as mock_close:
            
            result = render_matplotlib_chart(fig)
            
            assert result is True
            mock_pyplot.assert_called_once()
            mock_close.assert_called_once_with(fig)
    
    def test_concurrent_chart_rendering(self):
        """Test that multiple charts can be rendered without conflicts."""
        # Create multiple charts
        fig1, ax1 = plt.subplots()
        ax1.plot([1, 2, 3], [1, 4, 9])
        
        fig2 = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
        
        df = pd.DataFrame({'x': [1, 2, 3], 'y': [1, 4, 9]})
        
        charts = [fig1, fig2, df]
        
        # Test rendering all charts
        with patch('streamlit.pyplot') as mock_pyplot, \
             patch('streamlit.plotly_chart') as mock_plotly, \
             patch('streamlit.dataframe') as mock_dataframe:
            
            results = [render_chart(chart) for chart in charts]
            
            # All should succeed
            assert all(results)
            mock_pyplot.assert_called_once()
            mock_plotly.assert_called_once()
            mock_dataframe.assert_called_once()