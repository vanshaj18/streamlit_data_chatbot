"""
Visualization rendering components with caching and responsive sizing.
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
import hashlib
import pickle
from utils.error_handler import handle_viz_error, ErrorCategory


logger = logging.getLogger(__name__)

# Chart cache to store rendered charts
_CHART_CACHE = {}
_CACHE_MAX_SIZE = 50  # Maximum number of cached charts


def render_matplotlib_chart(fig: matplotlib.figure.Figure, 
                          title: Optional[str] = None,
                          use_container_width: bool = True,
                          cache_key: Optional[str] = None) -> bool:
    """
    Render matplotlib charts in Streamlit with caching and responsive sizing.
    
    Args:
        fig: Matplotlib figure object
        title: Optional title for the chart
        use_container_width: Whether to use full container width
        cache_key: Optional cache key for chart caching
        
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
        
        # Check cache first if cache_key is provided
        if cache_key and cache_key in _CHART_CACHE:
            cached_chart = _CHART_CACHE[cache_key]
            if title:
                st.subheader(title)
            st.image(cached_chart, use_column_width=use_container_width)
            logger.info("Rendered matplotlib chart from cache")
            return True
        
        # Add title if provided
        if title:
            st.subheader(title)
        
        # Configure figure for responsive display
        _configure_matplotlib_responsive(fig)
        
        # Render the chart
        st.pyplot(fig, use_container_width=use_container_width)
        
        # Cache the chart if cache_key is provided
        if cache_key:
            _cache_matplotlib_chart(fig, cache_key)
        
        # Clean up to prevent memory leaks
        plt.close(fig)
        
        logger.info("Successfully rendered matplotlib chart")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render matplotlib chart: {str(e)}")
        return False


def _configure_matplotlib_responsive(fig: matplotlib.figure.Figure) -> None:
    """
    Configure matplotlib figure for responsive display.
    
    Args:
        fig: Matplotlib figure to configure
    """
    try:
        # Set responsive figure size based on content
        fig.tight_layout(pad=2.0)
        
        # Configure for better mobile display
        fig.patch.set_facecolor('white')
        
        # Adjust font sizes for better readability
        for ax in fig.axes:
            ax.tick_params(labelsize=10)
            if ax.get_xlabel():
                ax.set_xlabel(ax.get_xlabel(), fontsize=12)
            if ax.get_ylabel():
                ax.set_ylabel(ax.get_ylabel(), fontsize=12)
            if ax.get_title():
                ax.set_title(ax.get_title(), fontsize=14, fontweight='bold')
        
        # Ensure legend fits properly
        for ax in fig.axes:
            legend = ax.get_legend()
            if legend:
                legend.set_bbox_to_anchor((1.05, 1), loc='upper left')
                
    except Exception as e:
        logger.warning(f"Failed to configure responsive matplotlib: {str(e)}")


def _cache_matplotlib_chart(fig: matplotlib.figure.Figure, cache_key: str) -> None:
    """
    Cache matplotlib chart as image bytes.
    
    Args:
        fig: Matplotlib figure to cache
        cache_key: Key to store the chart under
    """
    try:
        # Manage cache size
        if len(_CHART_CACHE) >= _CACHE_MAX_SIZE:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(_CHART_CACHE))
            del _CHART_CACHE[oldest_key]
        
        # Save figure to bytes
        buf = io.BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        
        # Store in cache
        _CHART_CACHE[cache_key] = buf.getvalue()
        
        logger.debug(f"Cached matplotlib chart with key: {cache_key}")
        
    except Exception as e:
        logger.warning(f"Failed to cache matplotlib chart: {str(e)}")


def render_plotly_chart(fig: Union[PlotlyFigure, go.Figure], 
                       title: Optional[str] = None,
                       use_container_width: bool = True,
                       height: Optional[int] = None,
                       cache_key: Optional[str] = None) -> bool:
    """
    Render plotly charts in Streamlit with responsive sizing and caching.
    
    Args:
        fig: Plotly figure object
        title: Optional title for the chart
        use_container_width: Whether to use full container width
        height: Optional height for the chart
        cache_key: Optional cache key for chart caching
        
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
        
        # Check cache first if cache_key is provided
        if cache_key and cache_key in _CHART_CACHE:
            cached_fig = _CHART_CACHE[cache_key]
            if title:
                st.subheader(title)
            st.plotly_chart(cached_fig, use_container_width=use_container_width, height=height)
            logger.info("Rendered plotly chart from cache")
            return True
        
        # Add title if provided
        if title:
            st.subheader(title)
        
        # Configure figure layout for responsive display
        _configure_plotly_responsive(fig, height)
        
        # Render the chart
        st.plotly_chart(
            fig, 
            use_container_width=use_container_width,
            height=height
        )
        
        # Cache the chart if cache_key is provided
        if cache_key:
            _cache_plotly_chart(fig, cache_key)
        
        logger.info("Successfully rendered plotly chart")
        return True
        
    except Exception as e:
        logger.error(f"Failed to render plotly chart: {str(e)}")
        return False


def _configure_plotly_responsive(fig: Union[PlotlyFigure, go.Figure], height: Optional[int] = None) -> None:
    """
    Configure plotly figure for responsive display.
    
    Args:
        fig: Plotly figure to configure
        height: Optional height constraint
    """
    try:
        # Responsive layout configuration
        responsive_layout = {
            'showlegend': True,
            'margin': dict(l=50, r=50, t=50, b=50),
            'font': dict(size=12),
            'autosize': True,
            'paper_bgcolor': 'white',
            'plot_bgcolor': 'white',
            'hovermode': 'closest'
        }
        
        # Add height if specified
        if height:
            responsive_layout['height'] = height
        
        # Configure for mobile responsiveness
        responsive_layout.update({
            'xaxis': dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickfont=dict(size=10)
            ),
            'yaxis': dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='lightgray',
                tickfont=dict(size=10)
            )
        })
        
        fig.update_layout(**responsive_layout)
        
        # Configure legend for better mobile display
        fig.update_layout(
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            )
        )
        
    except Exception as e:
        logger.warning(f"Failed to configure responsive plotly: {str(e)}")


def _cache_plotly_chart(fig: Union[PlotlyFigure, go.Figure], cache_key: str) -> None:
    """
    Cache plotly chart.
    
    Args:
        fig: Plotly figure to cache
        cache_key: Key to store the chart under
    """
    try:
        # Manage cache size
        if len(_CHART_CACHE) >= _CACHE_MAX_SIZE:
            # Remove oldest entry (simple FIFO)
            oldest_key = next(iter(_CHART_CACHE))
            del _CHART_CACHE[oldest_key]
        
        # Store figure in cache (plotly figures are JSON serializable)
        _CHART_CACHE[cache_key] = fig
        
        logger.debug(f"Cached plotly chart with key: {cache_key}")
        
    except Exception as e:
        logger.warning(f"Failed to cache plotly chart: {str(e)}")


def generate_chart_cache_key(chart_data: Any, title: Optional[str] = None, 
                            query: Optional[str] = None) -> str:
    """
    Generate a cache key for chart data.
    
    Args:
        chart_data: Chart data to generate key for
        title: Optional chart title
        query: Optional query that generated the chart
        
    Returns:
        str: Cache key
    """
    try:
        # Create a string representation of the chart data
        key_components = []
        
        if title:
            key_components.append(f"title:{title}")
        
        if query:
            key_components.append(f"query:{query}")
        
        # Add chart type
        key_components.append(f"type:{type(chart_data).__name__}")
        
        # For DataFrames, use shape and column names
        if isinstance(chart_data, pd.DataFrame):
            key_components.append(f"shape:{chart_data.shape}")
            key_components.append(f"cols:{','.join(chart_data.columns)}")
        
        # Create hash of the combined key
        key_string = "|".join(key_components)
        return hashlib.md5(key_string.encode()).hexdigest()
        
    except Exception as e:
        logger.warning(f"Failed to generate cache key: {str(e)}")
        return f"fallback_{hash(str(chart_data))}"


def render_chart(chart_data: Any, 
                title: Optional[str] = None,
                fallback_message: Optional[str] = None,
                query: Optional[str] = None,
                enable_caching: bool = True) -> bool:
    """
    Universal chart renderer that handles different chart types with caching.
    
    Args:
        chart_data: Chart data (matplotlib figure, plotly figure, or other)
        title: Optional title for the chart
        fallback_message: Custom fallback message for errors
        query: Optional query that generated the chart (for caching)
        enable_caching: Whether to enable chart caching
        
    Returns:
        bool: True if rendering successful, False otherwise
    """
    try:
        # Generate cache key if caching is enabled
        cache_key = None
        if enable_caching:
            cache_key = generate_chart_cache_key(chart_data, title, query)
        
        # Handle matplotlib figures
        if isinstance(chart_data, matplotlib.figure.Figure):
            return render_matplotlib_chart(chart_data, title, cache_key=cache_key)
        
        # Handle plotly figures
        elif isinstance(chart_data, (PlotlyFigure, go.Figure)):
            return render_plotly_chart(chart_data, title, cache_key=cache_key)
        
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


def clear_chart_cache() -> None:
    """Clear the chart cache to free memory."""
    global _CHART_CACHE
    _CHART_CACHE.clear()
    logger.info("Chart cache cleared")


def get_cache_stats() -> Dict[str, Any]:
    """
    Get chart cache statistics.
    
    Returns:
        Dict with cache statistics
    """
    return {
        'cache_size': len(_CHART_CACHE),
        'max_size': _CACHE_MAX_SIZE,
        'cache_keys': list(_CHART_CACHE.keys())
    }


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