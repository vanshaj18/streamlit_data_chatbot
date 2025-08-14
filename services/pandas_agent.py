"""
PandasAI agent wrapper for natural language query processing.
"""

import os
import logging
from typing import Any, Optional, Union, Dict
import pandas as pd
from dataclasses import dataclass
from datetime import datetime
from utils.error_handler import handle_query_error, ErrorCategory

# Monkey patch to fix the serialize_dataframe issue in PandasAI 3.0.0b19
def serialize_dataframe(self):
    """
    Serialize DataFrame for PandasAI templates.
    This is a workaround for the missing serialize_dataframe method in pandas.
    """
    # Return a string representation of the DataFrame for template rendering
    if len(self) > 5:
        # For large DataFrames, show head and tail
        head_str = self.head(3).to_string(index=False)
        tail_str = self.tail(2).to_string(index=False, header=False)
        return f"{head_str}\n...\n{tail_str}"
    else:
        # For small DataFrames, show all data
        return self.to_string(index=False)

# Monkey patch to fix the get_dialect issue in PandasAI 3.0.0b19
def get_dialect(self):
    """
    Get SQL dialect for DataFrame.
    This is a workaround for the missing get_dialect method in pandas.
    """
    return "duckdb"  # Default to duckdb dialect

# Monkey patch to fix the schema issue in PandasAI 3.0.0b19
class DataFrameSchema:
    """Mock schema object for DataFrame compatibility."""
    def __init__(self, dataframe):
        # ALWAYS use "df" as table name - no exceptions
        self.name = "df"
        self.table_name = "df"
        self._name = "df"
        self._table_name = "df"
        self.dataframe = dataframe
        
        # Add additional attributes that PandasAI might expect
        self.columns = list(dataframe.columns)
        self.dtypes = dataframe.dtypes.to_dict()
    
    def get_name(self):
        """Always return 'df' as table name."""
        return "df"
    
    def get_table_name(self):
        """Always return 'df' as table name."""
        return "df"
    
    @property
    def table(self):
        """Table name property - always returns 'df'."""
        return "df"

def get_schema(self):
    """
    Get schema for DataFrame.
    This is a workaround for the missing schema attribute in pandas.
    """
    return DataFrameSchema(self)

# Additional monkey patch to ensure table name is properly set
def get_table_name(self):
    """
    Get table name for DataFrame.
    This ensures PandasAI uses the correct table name in SQL queries.
    Always returns 'df' regardless of any other factors.
    """
    return "df"

# Additional method to ensure table name consistency
def table_name_property(self):
    """
    Alternative table name property.
    Always returns 'df' to ensure consistency across different PandasAI versions.
    """
    return "df"

# Method to override any existing table name attributes
def _get_table_name(self):
    """
    Internal method to get table name.
    Always returns 'df' to prevent any table name placeholder issues.
    """
    return "df"

# Additional methods that PandasAI might call
def get_name(self):
    """
    Get name for DataFrame.
    Always returns 'df' for table name consistency.
    """
    return "df"

def table_identifier(self):
    """
    Get table identifier for DataFrame.
    Always returns 'df' for SQL query consistency.
    """
    return "df"

# Add the methods to pandas DataFrame
pd.DataFrame.serialize_dataframe = serialize_dataframe
pd.DataFrame.get_dialect = get_dialect
pd.DataFrame.schema = property(get_schema)
pd.DataFrame.get_table_name = get_table_name
pd.DataFrame.table_name = property(table_name_property)
pd.DataFrame._get_table_name = _get_table_name
pd.DataFrame.get_name = get_name
pd.DataFrame.table_identifier = table_identifier

# Additional safeguards to ensure table name is always "df"
# Override any potential conflicting attributes
def _ensure_table_name_df(df):
    """Ensure DataFrame always uses 'df' as table name."""
    # Be careful with df.name as it can conflict with column names
    # Use custom attributes instead
    df._name = "df"
    df._table_name = "df"
    df._pandas_ai_table_name = "df"
    
    # Set additional attributes that might be used by PandasAI
    if hasattr(df, 'table'):
        df.table = "df"
    
    # Add a custom attribute to track our table name
    df._df_table_identifier = "df"
    
    return df

# Check if PandasAI is available
try:
    from pandasai import Agent
    try:
        # Try the newer import structure first
        from pandasai_litellm.litellm import LiteLLM
    except ImportError:
        try:
            # Fallback for different versions
            from pandasai.llm.litellm import LiteLLM
        except ImportError:
            from pandasai.llm import LiteLLM
    PANDASAI_AVAILABLE = True

    # Additional imports for deeper integration
    try:
        from pandasai.core.code_generation.code_cleaning import CodeCleaner
        from pandasai.query_builders.sql_parser import SQLParser
        ADVANCED_PANDASAI_AVAILABLE = True
    except ImportError:
        ADVANCED_PANDASAI_AVAILABLE = False

except ImportError:
    Agent = None
    LiteLLM = None
    PANDASAI_AVAILABLE = False
    ADVANCED_PANDASAI_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# Additional monkey patches for more aggressive table name handling
def _replace_table_name_placeholders(sql_query: str) -> str:
    """
    Replace any table name placeholders with 'df'.
    This is a fallback to ensure SQL queries always use the correct table name.
    """
    import re
    
    # Replace common table name placeholders
    replacements = [
        (r'<table_name>', 'df'),
        (r'<TABLE_NAME>', 'df'),
        (r'\{table_name\}', 'df'),
        (r'\{TABLE_NAME\}', 'df'),
        (r'sales_data', 'df'),  # Common incorrect table names
        (r'data_table', 'df'),
        (r'table1', 'df'),
        (r'main_table', 'df'),
    ]
    
    result = sql_query
    for pattern, replacement in replacements:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    
    return result

# Store original methods for monkey patching
original_agent_chat = None

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
            api_key: GEMINI_API_KEY. If not provided, will try to get from environment.
        """
        self.agent = None
        self.dataframe = None
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.logger = logging.getLogger(__name__)
        self.max_retries = 2
        
        if not PANDASAI_AVAILABLE:
            self.logger.warning("PandasAI is not available. Install with: pip install pandasai")
        
        if not self.api_key:
            self.logger.warning("GEMINI_API_KEY not found. Set GEMINI_API_KEY environment variable.")
    
    def initialize_agent(self, dataframe: pd.DataFrame, model_name: str) -> bool:
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
            self.logger.error("GEMINI_API_KEY is required")
            return False
            
        if dataframe is None or dataframe.empty:
            self.logger.error("DataFrame is None or empty")
            return False
            
        try:
            # Configure LiteLLM for Gemini API with proper authentication
            llm = LiteLLM(
                model=f'{model_name}',
                api_key=self.api_key,
                temperature=0.1,
                # Explicitly set the API base to avoid Google Cloud ADC issues
                api_base=None,  # Let LiteLLM use default Gemini API endpoint
                # Add custom headers to ensure proper authentication
                custom_llm_provider="gemini"
            )
            
            # Create a copy of the dataframe to avoid serialization issues
            df_copy = dataframe.copy()
            
            # Ensure the DataFrame has a consistent name for SQL generation using our safeguard function
            df = _ensure_table_name_df(df_copy)
            
            # Generate dynamic chart save path with file_name_plot_timestamp format
            from utils.session_manager import get_file_info
            file_info = get_file_info()
            
            if file_info and file_info.filename:
                # Extract filename without extension
                base_filename = file_info.filename.rsplit('.', 1)[0]
                # Create timestamp
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Create chart filename pattern
                chart_filename_pattern = f"{base_filename}_plot_{timestamp}"
                chart_save_path = f"exports/charts/{chart_filename_pattern}/"
            else:
                chart_save_path = "exports/charts/"
            
            # Create the Agent with configuration that avoids serialization issues
            config = {
                "llm": llm,
                "verbose": True,
                "enable_cache": False,  # Disable caching to avoid serialization issues
                "save_charts": True,
                "save_charts_path": chart_save_path,
                "custom_whitelisted_dependencies": ["matplotlib"], # no seaborn
                "open_charts": False,  # Don't auto-open charts
                "save_logs": False,  # Disable log saving to avoid serialization
                "response_parser": None,  # Disable response parser to avoid serialization
                "direct_sql": False,  # Disable direct SQL to avoid table name issues
                "enforce_privacy": False,  # Allow access to data for analysis
                "use_error_correction_framework": True  # Enable error correction
            }
            
            # Enhanced description to help with SQL generation
            description = """You are a data analyst agent. Your main goal is to help non-technical users to analyze data.
            - always use df as table name.
            - Never use placeholder names like <TABLE_NAME>.
            
            The schema of the table called 'df' is: 
            """ + ", ".join(df.columns.tolist())
            
            self.agent = Agent(df, 
                            description=description, 
                            config=config)
                            
            # Store original chat method for potential fallback
            global original_agent_chat
            if original_agent_chat is None and hasattr(self.agent, 'chat'):
                original_agent_chat = self.agent.chat
                            
            self.dataframe = df
            
            self.logger.info(f"PandasAI agent initialized with DataFrame shape: {dataframe.shape}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize PandasAI agent: {str(e)}")
            # Log the full traceback for debugging
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            return False
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess query to make it more explicit about table names and structure.
        
        Args:
            query: Original user query
            
        Returns:
            Enhanced query with explicit table name instructions
        """
        # For now, let's use a simpler approach to avoid logging issues
        if self.dataframe is not None:
            columns_info = ', '.join(self.dataframe.columns.tolist())
            # Simple enhancement that's less likely to cause logging issues
            enhanced_query = f"Using the table 'df' with columns ({columns_info}), {query}. Always use 'df' as the table name."
            return enhanced_query
        else:
            return query

    def process_query(self, query: str, timeout: int = 30) -> AgentResponse:
        """
        Execute natural language queries.
        
        Args:
            query: The natural language query to process
            timeout: Maximum time to wait for response in seconds
            
        Returns:
            AgentResponse: Structured response with content and metadata
            Error: 
        """
        start_time = datetime.now()
        
        if not self.agent:
            error_info = handle_query_error("Agent not initialized. Please upload data first.", query, show_ui=False)
            return AgentResponse(
                content=None,
                response_type="error",
                execution_time=0.0,
                timestamp=start_time,
                error_message=error_info.user_message
            )
        
        if not query or not query.strip():
            error_info = handle_query_error("Query cannot be empty.", query, show_ui=False)
            return AgentResponse(
                content=None,
                response_type="error",
                execution_time=0.0,
                timestamp=start_time,
                error_message=error_info.user_message
            )
        
        try:
            # Process the query with timeout handling
            self.logger.info(f"Processing query: {query}")
            self.logger.info(f"Question: {query}")
            self.logger.info(f"Running PandasAI with litellm LLM...")
            
            # Execute the query with error handling for serialization and SQL issues
            try:
                response = self.agent.chat(query)
                self.logger.info(f"Query executed successfully")
            except AttributeError as attr_error:
                if "serialize_dataframe" in str(attr_error):
                    # Handle the specific serialization error
                    self.logger.error(f"Serialization error detected: {attr_error}")
                    # Try to reinitialize the agent
                    if self.initialize_agent(self.dataframe):
                        response = self.agent.chat(query)
                    else:
                        raise RuntimeError("Failed to reinitialize agent after serialization error")
                else:
                    raise attr_error
            except Exception as general_error:
                # Handle other potential errors
                error_str = str(general_error)
                if "serialize_dataframe" in error_str or "pickle" in error_str.lower():
                    self.logger.error(f"Serialization-related error: {general_error}")
                    # Try to reinitialize the agent
                    if self.initialize_agent(self.dataframe):
                        response = self.agent.chat(query)
                    else:
                        raise RuntimeError("Failed to reinitialize agent after serialization error")
                elif ("<TABLE_NAME>" in error_str or 
                      "Expected table name" in error_str or
                      "<table_name>" in error_str or
                      "unauthorized table" in error_str):
                    # Handle SQL table name issues with multiple retry strategies
                    self.logger.error(f"SQL table name error detected: {general_error}")
                    
                    # Strategy 1: Try with explicit table name instruction
                    try:
                        explicit_query = f"""
                        Please analyze the data in the table named 'df'. The query is: {query}
                        Important: Always use 'df' as the table name in any SQL queries you generate.
                        Available columns: {', '.join(self.dataframe.columns.tolist())}
                        """
                        response = self.agent.chat(explicit_query)
                    except Exception as retry_error:
                        # Strategy 2: Try with pandas operations instead of SQL
                        try:
                            pandas_query = f"""
                            Using pandas operations on the DataFrame (not SQL), please {query}.
                            The DataFrame has columns: {', '.join(self.dataframe.columns.tolist())}
                            """
                            response = self.agent.chat(pandas_query)
                        except Exception as final_error:
                            # Strategy 3: Reinitialize and try again
                            if self.initialize_agent(self.dataframe):
                                response = self.agent.chat(f"Using the dataframe 'df', {query}")
                            else:
                                raise RuntimeError("Failed to resolve table name issue after multiple attempts")
                elif ("no code found" in error_str.lower() or 
                      "nocodefound" in error_str.lower()):
                    # Handle cases where LLM doesn't return properly formatted code
                    self.logger.error(f"No code found in LLM response: {general_error}")
                    
                    # Try with more explicit instructions for code generation
                    try:
                        code_explicit_query = f"""
                        Please provide Python code to answer: {query}
                        
                        Requirements:
                        - Use the DataFrame named 'df' 
                        - Available columns: {', '.join(self.dataframe.columns.tolist())}
                        - Always use 'df' as the table name in SQL queries
                        - Return your code in a proper Python code block
                        - Use the execute_sql_query function for SQL operations
                        
                        Format your response with:
                        ```python
                        # Your code here
                        ```
                        """
                        response = self.agent.chat(code_explicit_query)
                    except Exception as code_retry_error:
                        # Fallback: try a simpler approach
                        try:
                            simple_query = f"Calculate {query} using the dataframe 'df'"
                            response = self.agent.chat(simple_query)
                        except Exception as simple_error:
                            raise RuntimeError(f"Failed to get valid code response after multiple attempts: {simple_error}")
                else:
                    raise general_error
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Determine response type and handle accordingly
            return self.handle_response(response, execution_time, start_time)
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_info = handle_query_error(e, query, show_ui=False)
            
            self.logger.error(f"Query processing failed: {error_info.message}")
            self.logger.error(f"An error occurred during code generation: {str(e)}")
            self.logger.error(f"Stack Trace:")
            import traceback
            self.logger.error(traceback.format_exc())
            
            return AgentResponse(
                content=None,
                response_type="error",
                execution_time=execution_time,
                timestamp=start_time,
                error_message=error_info.user_message
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
            'created', 'generated', 'displayed', 'show', 'dispaly', 
            'generate', 'create', 'map', 'draw'
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
            return "API key issue. Please check your GEMINI_API_KEY configuration."
        
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
        
        if "<table_name>" in error_lower or "expected table name" in error_lower:
            return "There was an issue with data table access. Please try rephrasing your question or restart the session."
        
        if "sql" in error_lower and ("syntax" in error_lower or "error" in error_lower):
            return "There was an issue processing your query. Please try asking your question in a different way."
        
        if "no code found" in error_lower or "nocodefound" in error_lower:
            return "The AI couldn't generate proper code for your query. Please try rephrasing your question more clearly."
        
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

        if self.api_key is None or self.dataframe is None or self.agent is None:
            self.logger.error(f"Error: Either API key is missing or dataframe is not loading or something with agent")

        return (
            PANDASAI_AVAILABLE and 
            self.agent is not None and 
            self.dataframe is not None and 
            self.api_key is not None
        )