"""
Visualization rendering components.
"""

import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.figure
import plotly.graph_objects as go
import plotly.express as px
from plotly.graph_objs import Figure as PlotlyFigure
import pandas as pd
import logging
from typing import Any, Optional, Union, Dict, Tuple
import io
import base64
from utils.error_handler import handle_viz_error, ErrorCategory


logger = logging.getLogger(__name__)


def render_matplotlib_chart(fig: matplotlib.figure.Figure, 
                          title: Optional[str] = None,
                          use_container_width: bool = True) -> bool:
    """
    Render matplotlib charts in Streamlit.
    
    Args:
        fig: Matplotlib figure object
        title: Optional title for the chart
        use_container_width: Whether to use full container width
        
    Returns:
        bool: True if rendering successful, False otherwise
    """
    try:
        if fig is None:
            logger.error("Matplotlib figure is None")
            return False
            
        # Validate that it's a matplotlib figure
        if not isinstance(fig, matplotlib.figure.Figure):
            logger.error(f"Expected matplotlib Figure, got {type(fig)}")
            return False
        
        # Add title if provided
        if title:
            st.subheader(title)
        
        # Configure figure for better display
        fig.tight_layout()
        
        # Render the chart
        st.pyplot(fig, use_container_width=use_container_width)
        
        # Clean up to prevent memory leaks
        plt.close(fig)
        
        logger.info("Successfully rendered matplotlib chart")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render matplotlib chart: {str(e)}")
        return False


def render_plotly_chart(fig: Union[PlotlyFigure, go.Figure], 
                       title: Optional[str] = None,
                       use_container_width: bool = True,
                       height: Optional[int] = None) -> bool:
    """
    Render plotly charts in Streamlit.
    
    Args:
        fig: Plotly figure object
        title: Optional title for the chart
        use_container_width: Whether to use full container width
        height: Optional height for the chart
        
    Returns:
        bool: True if rendering successful, False otherwise
    """
    try:
        if fig is None:
            logger.error("Plotly figure is None")
            return False
            
        # Validate that it's a plotly figure
        if not isinstance(fig, (PlotlyFigure, go.Figure)):
            logger.error(f"Expected Plotly Figure, got {type(fig)}")
            return False
        
        # Add title if provided
        if title:
            st.subheader(title)
        
        # Configure figure layout for better display
        fig.update_layout(
            showlegend=True,
            margin=dict(l=0, r=0, t=30, b=0),
            font=dict(size=12)
        )
        
        # Render the chart
        st.plotly_chart(
            fig, 
            use_container_width=use_container_width,
            height=height
        )
        
        logger.info("Successfully rendered plotly chart")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render plotly chart: {str(e)}")
        return False


def render_chart(chart_data: Any, 
                title: Optional[str] = None,
                fallback_message: Optional[str] = None) -> bool:
    """
    Universal chart renderer that handles different chart types.
    
    Args:
        chart_data: Chart data (matplotlib figure, plotly figure, or other)
        title: Optional title for the chart
        fallback_message: Custom fallback message for errors
        
    Returns:
        bool: True if rendering successful, False otherwise
    """
    try:
        # Handle matplotlib figures
        if isinstance(chart_data, matplotlib.figure.Figure):
            return render_matplotlib_chart(chart_data, title)
        
        # Handle plotly figures
        elif isinstance(chart_data, (PlotlyFigure, go.Figure)):
            return render_plotly_chart(chart_data, title)
        
        # Handle pandas DataFrames as tables
        elif isinstance(chart_data, pd.DataFrame):
            if title:
                st.subheader(title)
            st.dataframe(chart_data, use_container_width=True)
            logger.info("Successfully rendered DataFrame as table")
            return True
        
        # Handle raw image data or base64 encoded images
        elif isinstance(chart_data, (bytes, str)):
            return _render_image_data(chart_data, title)
        
        # Handle other chart objects that might have a 'show' method
        elif hasattr(chart_data, 'show'):
            if title:
                st.subheader(title)
            st.plotly_chart(chart_data, use_container_width=True)
            logger.info("Successfully rendered chart with show() method")
            return True
        
        # Handle objects that might have a 'savefig' method (matplotlib-like)
        elif hasattr(chart_data, 'savefig'):
            try:
                if title:
                    st.subheader(title)
                st.pyplot(chart_data)
                logger.info("Successfully rendered chart with savefig() method")
                return True
            except Exception as e:
                logger.warning(f"Failed to render chart with savefig method: {str(e)}")
                return _render_fallback_content(chart_data, title, fallback_message)
        
        # Fallback: try to display as text or JSON
        else:
            logger.warning(f"Unknown chart type: {type(chart_data)}")
            return _render_fallback_content(chart_data, title, fallback_message)
            
    except Exception as e:
        logger.error(f"Error in universal chart renderer: {str(e)}")
        return handle_chart_error(str(e), fallback_message)


def _render_image_data(image_data: Union[bytes, str], title: Optional[str] = None) -> bool:
    """
    Render image data (bytes or base64 string).
    
    Args:
        image_data: Image data as bytes or base64 string
        title: Optional title for the image
        
    Returns:
        bool: True if rendering successful, False otherwise
    """
    try:
        if title:
            st.subheader(title)
            
        if isinstance(image_data, str):
            # Handle base64 encoded images
            if image_data.startswith('data:image'):
                st.markdown(f'<img src="{image_data}" style="max-width: 100%;">', 
                          unsafe_allow_html=True)
            else:
                # Try to decode base64
                image_bytes = base64.b64decode(image_data)
                st.image(image_bytes, use_column_width=True)
        else:
            # Handle raw bytes
            st.image(image_data, use_column_width=True)
            
        logger.info("Successfully rendered image data")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render image data: {str(e)}")
        return False


def _render_fallback_content(content: Any, title: Optional[str] = None, 
                           fallback_message: Optional[str] = None) -> bool:
    """
    Render content as fallback when chart rendering fails.
    
    Args:
        content: Content to render
        title: Optional title
        fallback_message: Custom fallback message
        
    Returns:
        bool: True if fallback rendering successful, False otherwise
    """
    try:
        if title:
            st.subheader(title)
            
        if fallback_message:
            st.info(fallback_message)
        else:
            st.info("Chart could not be rendered. Displaying raw content:")
        
        # Try to display content in a readable format
        if hasattr(content, '__dict__'):
            st.json(content.__dict__)
        elif isinstance(content, (dict, list)):
            st.json(content)
        else:
            st.text(str(content))
            
        logger.info("Successfully rendered fallback content")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render fallback content: {str(e)}")
        return False


def handle_chart_error(error_message: str, 
                      custom_message: Optional[str] = None) -> bool:
    """
    Handle visualization errors with user-friendly messages.
    
    Args:
        error_message: The original error message
        custom_message: Custom error message to display
        
    Returns:
        bool: Always returns False to indicate failure
    """
    try:
        # Use centralized error handler
        error_info = handle_viz_error(error_message, None, show_ui=True)
        
        logger.error(f"Chart error handled: {error_message}")
        return False
        
    except Exception as e:
        logger.error(f"Error in error handler: {str(e)}")
        st.error("An unexpected error occurred while displaying the chart.")
        return False


def _get_user_friendly_error_message(error_message: str, 
                                   custom_message: Optional[str] = None) -> str:
    """
    Convert technical error messages to user-friendly ones.
    
    Args:
        error_message: Original technical error message
        custom_message: Custom message to use instead
        
    Returns:
        str: User-friendly error message
    """
    if custom_message:
        return custom_message
    
    error_lower = error_message.lower()
    
    # Common error patterns and user-friendly messages
    if "memory" in error_lower or "out of memory" in error_lower:
        return "Not enough memory to create this visualization. Try with a smaller dataset."
    
    if "column" in error_lower and ("not found" in error_lower or "missing" in error_lower):
        return "One or more columns referenced in your query don't exist in your data."
    
    if "empty" in error_lower or "no data" in error_lower:
        return "No data available to create the visualization."
    
    if "type" in error_lower and "error" in error_lower:
        return "Data type mismatch. Check that your data is in the expected format."
    
    if "matplotlib" in error_lower or "pyplot" in error_lower:
        return "Error creating chart with matplotlib. Try asking for a different type of visualization."
    
    if "plotly" in error_lower:
        return "Error creating interactive chart. Try asking for a simpler visualization."
    
    if "figure" in error_lower:
        return "Could not generate the requested chart. Try rephrasing your question."
    
    # Generic fallback
    return "Unable to create the visualization. Please try a different approach or simpler query."


def create_sample_charts() -> Dict[str, Any]:
    """
    Create sample charts for testing purposes.
    
    Returns:
        Dict containing sample matplotlib and plotly charts
    """
    # Sample data
    data = pd.DataFrame({
        'x': range(10),
        'y': [i**2 for i in range(10)],
        'category': ['A', 'B'] * 5
    })
    
    # Create matplotlib chart
    fig_mpl, ax = plt.subplots(figsize=(8, 6))
    ax.plot(data['x'], data['y'], marker='o')
    ax.set_title('Sample Matplotlib Chart')
    ax.set_xlabel('X Values')
    ax.set_ylabel('Y Values')
    
    # Create plotly chart
    fig_plotly = px.scatter(data, x='x', y='y', color='category', 
                           title='Sample Plotly Chart')
    
    return {
        'matplotlib': fig_mpl,
        'plotly': fig_plotly,
        'dataframe': data
    }


def validate_chart_data(chart_data: Any) -> Tuple[bool, str]:
    """
    Validate chart data before rendering.
    
    Args:
        chart_data: Data to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if chart_data is None:
            return False, "Chart data is None"
        
        # Check for matplotlib figures
        if isinstance(chart_data, matplotlib.figure.Figure):
            if len(chart_data.axes) == 0:
                return False, "Matplotlib figure has no axes"
            return True, ""
        
        # Check for plotly figures
        if isinstance(chart_data, (PlotlyFigure, go.Figure)):
            if not chart_data.data:
                return False, "Plotly figure has no data"
            return True, ""
        
        # Check for DataFrames
        if isinstance(chart_data, pd.DataFrame):
            if chart_data.empty:
                return False, "DataFrame is empty"
            return True, ""
        
        # For other types, assume valid if not None
        return True, ""
        
    except Exception as e:
        return False, f"Validation error: {str(e)}"