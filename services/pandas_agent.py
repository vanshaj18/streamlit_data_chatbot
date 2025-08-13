"""
PandasAI agent wrapper for natural language query processing.
"""

import os
import logging
from typing import Any, Optional, Union, Dict
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

try:
    from pandasai import Agent
    from pandasai.llm import GooglePalm
    from pandasai.responses.response_parser import ResponseParser
    PANDASAI_AVAILABLE = True
except (ImportError, TypeError, SyntaxError) as e:
    # Handle import errors, including Python version compatibility issues
    PANDASAI_AVAILABLE = False
    Agent = None
    ResponseParser = None
    GooglePalm = None
    ResponseParser = None


@dataclass
class AgentResponse:
    """Structured response from PandasAI agent."""
    content: Any
    response_type: str  # "text", "plot", "dataframe", "error"
    execution_time: float
    timestamp: datetime
    error_message: Optional[str] = None


class PandasAgent:
    """Wrapper for PandasAI agent functionality."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the PandasAI agent.
        
        Args:
            api_key: GooglePalm API key. If not provided, will try to get from environment.
        """
        self.agent = None
        self.dataframe = None
        self.api_key = api_key or os.getenv('GooglePalm_API_KEY')
        self.logger = logging.getLogger(__name__)
        
        if not PANDASAI_AVAILABLE:
            self.logger.warning("PandasAI is not available. Install with: pip install pandasai")
        
        if not self.api_key:
            self.logger.warning("GooglePalm API key not found. Set GooglePalm_API_KEY environment variable.")
    
    def initialize_agent(self, dataframe: pd.DataFrame) -> bool:
        """
        Set up PandasAI with DataFrame.
        
        Args:
            dataframe: The pandas DataFrame to analyze
            
        Returns:
            bool: True if initialization successful, False otherwise
        """
        if not PANDASAI_AVAILABLE:
            self.logger.error("PandasAI is not available")
            return False
            
        if not self.api_key:
            self.logger.error("GooglePalm API key is required")
            return False
            
        if dataframe is None or dataframe.empty:
            self.logger.error("DataFrame is None or empty")
            return False
            
        try:
            # Initialize the LLM
            llm = GooglePalm(api_token=self.api_key)
            
            # Create the agent with the DataFrame
            self.agent = Agent(dataframe, config={"llm": llm, "verbose": False})
            self.dataframe = dataframe
            
            self.logger.info(f"PandasAI agent initialized with DataFrame shape: {dataframe.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PandasAI agent: {str(e)}")
            return False
    
    def process_query(self, query: str, timeout: int = 30) -> AgentResponse:
        """
        Execute natural language queries.
        
        Args:
            query: The natural language query to process
            timeout: Maximum time to wait for response in seconds
            
        Returns:
            AgentResponse: Structured response with content and metadata
        """
        start_time = datetime.now()
        
        if not self.agent:
            return AgentResponse(
                content=None,
                response_type="error",
                execution_time=0.0,
                timestamp=start_time,
                error_message="Agent not initialized. Please upload data first."
            )
        
        if not query or not query.strip():
            return AgentResponse(
                content=None,
                response_type="error",
                execution_time=0.0,
                timestamp=start_time,
                error_message="Query cannot be empty."
            )
        
        try:
            # Process the query with timeout handling
            self.logger.info(f"Processing query: {query}")
            
            # Execute the query
            response = self.agent.chat(query)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Determine response type and handle accordingly
            return self.handle_response(response, execution_time, start_time)
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = self._format_error_message(str(e))
            
            self.logger.error(f"Query processing failed: {error_msg}")
            
            return AgentResponse(
                content=None,
                response_type="error",
                execution_time=execution_time,
                timestamp=start_time,
                error_message=error_msg
            )
    
    def handle_response(self, response: Any, execution_time: float, timestamp: datetime) -> AgentResponse:
        """
        Process and format agent responses with enhanced chart detection.
        
        Args:
            response: Raw response from PandasAI
            execution_time: Time taken to execute the query
            timestamp: When the query was started
            
        Returns:
            AgentResponse: Formatted response with type classification
        """
        try:
            # Handle different response types
            if response is None:
                return AgentResponse(
                    content="No response generated. Try rephrasing your question.",
                    response_type="text",
                    execution_time=execution_time,
                    timestamp=timestamp
                )
            
            # Enhanced chart detection
            if self._is_chart_response(response):
                return AgentResponse(
                    content=response,
                    response_type="plot",
                    execution_time=execution_time,
                    timestamp=timestamp
                )
            
            # Check if response is a DataFrame
            if isinstance(response, pd.DataFrame):
                # For small DataFrames, treat as plot (table visualization)
                # For large DataFrames, might want to show summary
                response_type = "plot" if len(response) <= 100 else "dataframe"
                return AgentResponse(
                    content=response,
                    response_type=response_type,
                    execution_time=execution_time,
                    timestamp=timestamp
                )
            
            # Handle string responses that might contain chart references
            if isinstance(response, str) and self._contains_chart_reference(response):
                return AgentResponse(
                    content=response,
                    response_type="text",
                    execution_time=execution_time,
                    timestamp=timestamp
                )
            
            # Default to text response
            return AgentResponse(
                content=str(response),
                response_type="text",
                execution_time=execution_time,
                timestamp=timestamp
            )
            
        except Exception as e:
            self.logger.error(f"Error handling response: {str(e)}")
            return AgentResponse(
                content=None,
                response_type="error",
                execution_time=execution_time,
                timestamp=timestamp,
                error_message=f"Error processing response: {str(e)}"
            )
    
    def _is_chart_response(self, response: Any) -> bool:
        """
        Enhanced detection of chart/plot responses.
        
        Args:
            response: Response to check
            
        Returns:
            bool: True if response appears to be a chart
        """
        try:
            # Check for matplotlib figures
            if hasattr(response, 'savefig') or hasattr(response, 'figure'):
                return True
            
            # Check for plotly figures
            if hasattr(response, 'show') or hasattr(response, 'to_html'):
                return True
            
            # Check type string for matplotlib/plotly references
            response_type_str = str(type(response)).lower()
            if any(chart_lib in response_type_str for chart_lib in 
                   ['matplotlib', 'plotly', 'figure', 'axes']):
                return True
            
            # Check for common chart object attributes
            chart_attributes = ['data', 'layout', 'axes', 'gca']
            if any(hasattr(response, attr) for attr in chart_attributes):
                # Additional validation to avoid false positives
                if hasattr(response, 'data') and hasattr(response, 'layout'):
                    return True  # Likely plotly
                if hasattr(response, 'axes') and hasattr(response, 'gca'):
                    return True  # Likely matplotlib
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Error in chart detection: {str(e)}")
            return False
    
    def _contains_chart_reference(self, text: str) -> bool:
        """
        Check if text response contains references to charts that were created.
        
        Args:
            text: Text response to check
            
        Returns:
            bool: True if text mentions chart creation
        """
        chart_keywords = [
            'chart', 'plot', 'graph', 'visualization', 'figure',
            'histogram', 'scatter', 'bar chart', 'line plot',
            'created', 'generated', 'displayed'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in chart_keywords)
    
    def _format_error_message(self, error: str) -> str:
        """
        Format error messages to be user-friendly.
        
        Args:
            error: Raw error message
            
        Returns:
            str: User-friendly error message
        """
        error_lower = error.lower()
        
        # Common error patterns and user-friendly messages
        if "api" in error_lower and ("key" in error_lower or "token" in error_lower):
            return "API key issue. Please check your GooglePalm API key configuration."
        
        if "timeout" in error_lower or "time" in error_lower:
            return "Query is taking too long. Please try a simpler question."
        
        if "rate limit" in error_lower:
            return "Too many requests. Please wait a moment and try again."
        
        if "invalid" in error_lower and "query" in error_lower:
            return "Invalid query. Please rephrase your question more clearly."
        
        if "column" in error_lower and "not found" in error_lower:
            return "Column not found in your data. Please check the column names and try again."
        
        if "memory" in error_lower or "out of memory" in error_lower:
            return "Not enough memory to process this query. Try working with a smaller dataset."
        
        # Generic fallback
        return "Unable to process your query. Please try rephrasing your question."
    
    def get_dataframe_info(self) -> Optional[Dict[str, Any]]:
        """
        Get information about the current DataFrame.
        
        Returns:
            Dict with DataFrame metadata or None if no DataFrame loaded
        """
        if self.dataframe is None:
            return None
            
        try:
            return {
                "shape": self.dataframe.shape,
                "columns": list(self.dataframe.columns),
                "dtypes": self.dataframe.dtypes.to_dict(),
                "memory_usage": self.dataframe.memory_usage(deep=True).sum(),
                "null_counts": self.dataframe.isnull().sum().to_dict()
            }
        except Exception as e:
            self.logger.error(f"Error getting DataFrame info: {str(e)}")
            return None
    
    def is_initialized(self) -> bool:
        """
        Check if the agent is properly initialized.
        
        Returns:
            bool: True if agent is ready to process queries
        """
        return (
            PANDASAI_AVAILABLE and 
            self.agent is not None and 
            self.dataframe is not None and 
            self.api_key is not None
        )