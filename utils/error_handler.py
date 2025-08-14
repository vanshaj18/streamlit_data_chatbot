"""
Centralized error handling system for the Data Chatbot Dashboard.
"""

import logging
import traceback
import streamlit as st
from typing import Optional, Dict, Any, Callable, Union
from enum import Enum
from dataclasses import dataclass
from datetime import datetime


class ErrorCategory(Enum):
    """Categories of errors that can occur in the application."""
    FILE_UPLOAD = "file_upload"
    QUERY_PROCESSING = "query_processing"
    VISUALIZATION = "visualization"
    PLOTTING = "plotting"
    SESSION_MANAGEMENT = "session_management"
    API_ERROR = "api_error"
    SYSTEM_ERROR = "system_error"


class ErrorSeverity(Enum):
    """Severity levels for errors."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorInfo:
    """Structured error information."""
    category: ErrorCategory
    severity: ErrorSeverity
    message: str
    user_message: str
    suggestions: list[str]
    timestamp: datetime
    technical_details: Optional[str] = None
    recovery_action: Optional[Callable] = None


class PlottingErrorType(Enum):
    """Specific types of plotting errors."""
    MATPLOTLIB_ERROR = "matplotlib_error"
    PANDASAI_PLOTTING = "pandasai_plotting"
    DATA_COMPATIBILITY = "data_compatibility"
    CHART_GENERATION = "chart_generation"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    TIMEOUT = "timeout"
    INVALID_CHART_TYPE = "invalid_chart_type"


@dataclass
class PlottingErrorInfo(ErrorInfo):
    """Extended error information for plotting-specific errors."""
    plotting_error_type: PlottingErrorType = PlottingErrorType.CHART_GENERATION
    chart_type: Optional[str] = None
    data_columns: Optional[list[str]] = None
    fallback_available: bool = False
    suggested_chart_type: Optional[str] = None


class ErrorHandler:
    """Centralized error handler with user-friendly messages and recovery suggestions."""
    
    def __init__(self):
        """Initialize the error handler."""
        self.logger = logging.getLogger(__name__)
        self._error_patterns = self._initialize_error_patterns()
        self._recovery_suggestions = self._initialize_recovery_suggestions()
    
    def handle_error(self, 
                    error: Union[Exception, str],
                    category: ErrorCategory,
                    context: Optional[Dict[str, Any]] = None,
                    show_ui: bool = True) -> ErrorInfo:
        """
        Handle an error with appropriate user messaging and logging.
        
        Args:
            error: The error to handle (Exception or string)
            category: Category of the error
            context: Additional context information
            show_ui: Whether to display UI error message
            
        Returns:
            ErrorInfo: Structured error information
        """
        # Extract error message
        if isinstance(error, Exception):
            error_message = str(error)
            technical_details = traceback.format_exc()
        else:
            error_message = error
            technical_details = None
        
        # Determine severity and create error info
        severity = self._determine_severity(error_message, category)
        error_info = self._create_error_info(
            error_message, category, severity, technical_details, context
        )
        
        # Log the error
        self._log_error(error_info)
        
        # Display UI message if requested
        if show_ui:
            self._display_error_ui(error_info)
        
        return error_info
    
    def handle_file_upload_error(self, error: Union[Exception, str], 
                                filename: Optional[str] = None,
                                show_ui: bool = True) -> ErrorInfo:
        """
        Handle file upload specific errors.
        
        Args:
            error: The error to handle
            filename: Name of the file that caused the error
            
        Returns:
            ErrorInfo: Structured error information
        """
        context = {"filename": filename} if filename else None
        return self.handle_error(error, ErrorCategory.FILE_UPLOAD, context, show_ui)
    
    def handle_query_processing_error(self, error: Union[Exception, str],
                                    query: Optional[str] = None,
                                    show_ui: bool = True) -> ErrorInfo:
        """
        Handle query processing specific errors.
        
        Args:
            error: The error to handle
            query: The query that caused the error
            
        Returns:
            ErrorInfo: Structured error information
        """
        context = {"query": query} if query else None
        return self.handle_error(error, ErrorCategory.QUERY_PROCESSING, context, show_ui)
    
    def handle_visualization_error(self, error: Union[Exception, str],
                                 chart_type: Optional[str] = None,
                                 show_ui: bool = True) -> ErrorInfo:
        """
        Handle visualization specific errors.
        
        Args:
            error: The error to handle
            chart_type: Type of chart that failed to render
            
        Returns:
            ErrorInfo: Structured error information
        """
        context = {"chart_type": chart_type} if chart_type else None
        return self.handle_error(error, ErrorCategory.VISUALIZATION, context, show_ui)
    
    def handle_plotting_error(self, error: Union[Exception, str],
                            chart_type: Optional[str] = None,
                            data_columns: Optional[list[str]] = None,
                            query: Optional[str] = None,
                            show_ui: bool = True) -> PlottingErrorInfo:
        """
        Handle plotting-specific errors with enhanced classification and recovery.
        
        Args:
            error: The error to handle
            chart_type: Type of chart that failed
            data_columns: Available data columns
            query: Original user query
            show_ui: Whether to display UI error message
            
        Returns:
            PlottingErrorInfo: Enhanced plotting error information
        """
        # Extract error message
        if isinstance(error, Exception):
            error_message = str(error)
            technical_details = traceback.format_exc()
        else:
            error_message = error
            technical_details = None
        
        # Classify plotting error type
        plotting_error_type = self._classify_plotting_error(error_message)
        
        # Determine severity
        severity = self._determine_plotting_severity(error_message, plotting_error_type)
        
        # Create context
        context = {
            "chart_type": chart_type,
            "data_columns": data_columns,
            "query": query,
            "plotting_error_type": plotting_error_type.value
        }
        
        # Get user-friendly message and suggestions
        user_message = self._get_plotting_user_message(error_message, plotting_error_type, chart_type)
        suggestions = self._get_plotting_recovery_suggestions(
            error_message, plotting_error_type, chart_type, data_columns, query
        )
        
        # Check if fallback is available
        fallback_available = self._is_fallback_available(plotting_error_type, chart_type)
        suggested_chart_type = self._suggest_alternative_chart_type(
            plotting_error_type, chart_type, data_columns
        )
        
        # Create plotting error info
        plotting_error_info = PlottingErrorInfo(
            category=ErrorCategory.PLOTTING,
            severity=severity,
            message=error_message,
            user_message=user_message,
            suggestions=suggestions,
            timestamp=datetime.now(),
            technical_details=technical_details,
            plotting_error_type=plotting_error_type,
            chart_type=chart_type,
            data_columns=data_columns,
            fallback_available=fallback_available,
            suggested_chart_type=suggested_chart_type
        )
        
        # Log the error
        self._log_error(plotting_error_info)
        
        # Display UI message if requested
        if show_ui:
            self._display_plotting_error_ui(plotting_error_info)
        
        return plotting_error_info
    
    def _create_error_info(self, error_message: str, category: ErrorCategory,
                          severity: ErrorSeverity, technical_details: Optional[str],
                          context: Optional[Dict[str, Any]]) -> ErrorInfo:
        """Create structured error information."""
        user_message = self._get_user_friendly_message(error_message, category)
        suggestions = self._get_recovery_suggestions(error_message, category, context)
        
        return ErrorInfo(
            category=category,
            severity=severity,
            message=error_message,
            user_message=user_message,
            suggestions=suggestions,
            timestamp=datetime.now(),
            technical_details=technical_details
        )
    
    def _determine_severity(self, error_message: str, category: ErrorCategory) -> ErrorSeverity:
        """Determine error severity based on message and category."""
        error_lower = error_message.lower()
        
        # Critical errors
        if any(keyword in error_lower for keyword in 
               ["memory", "out of memory", "system", "critical", "fatal"]):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if any(keyword in error_lower for keyword in 
               ["api key", "authentication", "permission", "access denied"]):
            return ErrorSeverity.HIGH
        
        # Medium severity errors
        if any(keyword in error_lower for keyword in 
               ["timeout", "connection", "network", "rate limit"]):
            return ErrorSeverity.MEDIUM
        
        # Category-based severity
        if category == ErrorCategory.FILE_UPLOAD:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.QUERY_PROCESSING:
            return ErrorSeverity.MEDIUM
        elif category == ErrorCategory.VISUALIZATION:
            return ErrorSeverity.LOW
        elif category == ErrorCategory.PLOTTING:
            return ErrorSeverity.MEDIUM
        
        return ErrorSeverity.LOW
    
    def _get_user_friendly_message(self, error_message: str, 
                                 category: ErrorCategory) -> str:
        """Convert technical error messages to user-friendly ones."""
        error_lower = error_message.lower()
        
        # Check predefined patterns
        for pattern, friendly_message in self._error_patterns.items():
            if pattern in error_lower:
                return friendly_message
        
        # Category-specific fallbacks
        if category == ErrorCategory.FILE_UPLOAD:
            return "There was a problem with your file. Please check the file format and try again."
        elif category == ErrorCategory.QUERY_PROCESSING:
            return "Unable to process your query. Please try rephrasing your question."
        elif category == ErrorCategory.VISUALIZATION:
            return "Could not create the visualization. The data might not be suitable for this chart type."
        elif category == ErrorCategory.PLOTTING:
            return "There was a problem generating the chart. Please try a different visualization approach."
        elif category == ErrorCategory.API_ERROR:
            return "There was a problem connecting to the AI service. Please try again later."
        elif category == ErrorCategory.SESSION_MANAGEMENT:
            return "There was a problem with your session. Please refresh the page."
        else:
            return "An unexpected error occurred. Please try again."
    
    def _get_recovery_suggestions(self, error_message: str, category: ErrorCategory,
                                context: Optional[Dict[str, Any]]) -> list[str]:
        """Get recovery suggestions based on error and context."""
        error_lower = error_message.lower()
        suggestions = []
        
        # Error-specific suggestions
        if "memory" in error_lower:
            suggestions.extend([
                "Try using a smaller dataset",
                "Close other applications to free up memory",
                "Consider sampling your data before analysis"
            ])
        
        elif "api key" in error_lower or "authentication" in error_lower:
            suggestions.extend([
                "Check your API key configuration",
                "Verify your API key is valid and active",
                "Contact support if the problem persists"
            ])
        
        elif "timeout" in error_lower:
            suggestions.extend([
                "Try a simpler query",
                "Check your internet connection",
                "Wait a moment and try again"
            ])
        
        elif "column" in error_lower and "not found" in error_lower:
            suggestions.extend([
                "Check the column names in your data",
                "Use the data preview to see available columns",
                "Make sure your query references existing columns"
            ])
        
        # Category-specific suggestions
        category_suggestions = self._recovery_suggestions.get(category, [])
        suggestions.extend(category_suggestions)
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(suggestions))
    
    def _classify_plotting_error(self, error_message: str) -> PlottingErrorType:
        """
        Classify plotting errors into specific types for targeted handling.
        
        Args:
            error_message: The error message to classify
            
        Returns:
            PlottingErrorType: The classified error type
        """
        error_lower = error_message.lower()
        
        # Matplotlib-specific errors (check specific patterns first)
        if any(keyword in error_lower for keyword in [
            "matplotlib", "pyplot", "savefig", "no display name", 
            "backend", "tkinter", "qt"
        ]):
            return PlottingErrorType.MATPLOTLIB_ERROR
        
        # More specific matplotlib patterns
        if any(pattern in error_lower for pattern in [
            "figure creation", "axes"
        ]):
            return PlottingErrorType.MATPLOTLIB_ERROR
        
        # Check for plot but not in other contexts
        if " plot " in error_lower and "unknown" in error_lower:
            return PlottingErrorType.INVALID_CHART_TYPE
        
        # PandasAI plotting errors
        if any(keyword in error_lower for keyword in [
            "pandasai", "smartdataframe", "generate_code"
        ]):
            return PlottingErrorType.PANDASAI_PLOTTING
        
        # More specific PandasAI patterns
        if any(pattern in error_lower for pattern in [
            "base64 decode", "sql table name", "chat response"
        ]):
            return PlottingErrorType.PANDASAI_PLOTTING
        
        # Data compatibility errors
        if any(pattern in error_lower for pattern in [
            "column", "missing column", "data type", "dtype",
            "insufficient data", "empty dataframe", "no numeric data",
            "categorical data", "string cannot be converted"
        ]):
            return PlottingErrorType.DATA_COMPATIBILITY
        
        # Memory and resource errors
        if any(keyword in error_lower for keyword in [
            "memory", "out of memory", "memoryerror", "resource",
            "too large", "allocation failed"
        ]):
            return PlottingErrorType.MEMORY_EXHAUSTION
        
        # Timeout errors
        if any(keyword in error_lower for keyword in [
            "timeout", "timed out", "time limit", "took too long"
        ]):
            return PlottingErrorType.TIMEOUT
        
        # Invalid chart type errors
        if any(pattern in error_lower for pattern in [
            "invalid chart", "unsupported chart", "chart type",
            "unknown plot type", "invalid plot type", "chart type not recognized"
        ]):
            return PlottingErrorType.INVALID_CHART_TYPE
        
        # Default to general chart generation error
        return PlottingErrorType.CHART_GENERATION
    
    def _determine_plotting_severity(self, error_message: str, 
                                   plotting_error_type: PlottingErrorType) -> ErrorSeverity:
        """
        Determine severity for plotting-specific errors.
        
        Args:
            error_message: The error message
            plotting_error_type: The classified plotting error type
            
        Returns:
            ErrorSeverity: The determined severity level
        """
        # Memory errors are critical
        if plotting_error_type == PlottingErrorType.MEMORY_EXHAUSTION:
            return ErrorSeverity.CRITICAL
        
        # PandasAI and matplotlib errors are high priority
        if plotting_error_type in [PlottingErrorType.PANDASAI_PLOTTING, 
                                 PlottingErrorType.MATPLOTLIB_ERROR]:
            return ErrorSeverity.HIGH
        
        # Data compatibility and timeout are medium
        if plotting_error_type in [PlottingErrorType.DATA_COMPATIBILITY, 
                                 PlottingErrorType.TIMEOUT]:
            return ErrorSeverity.MEDIUM
        
        # Other plotting errors are low severity
        return ErrorSeverity.LOW
    
    def _get_plotting_user_message(self, error_message: str, 
                                 plotting_error_type: PlottingErrorType,
                                 chart_type: Optional[str] = None) -> str:
        """
        Generate user-friendly messages for plotting errors.
        
        Args:
            error_message: The original error message
            plotting_error_type: The classified error type
            chart_type: The requested chart type
            
        Returns:
            str: User-friendly error message
        """
        chart_desc = f" {chart_type}" if chart_type else ""
        
        if plotting_error_type == PlottingErrorType.MATPLOTLIB_ERROR:
            return f"There was a problem with the chart rendering engine while creating your{chart_desc} chart. This might be due to display or graphics issues."
        
        elif plotting_error_type == PlottingErrorType.PANDASAI_PLOTTING:
            return f"The AI had trouble generating the plotting code for your{chart_desc} chart. The generated code may have issues or incompatibilities."
        
        elif plotting_error_type == PlottingErrorType.DATA_COMPATIBILITY:
            return f"Your data isn't compatible with the requested{chart_desc} chart type. The data might be missing required columns or have incompatible data types."
        
        elif plotting_error_type == PlottingErrorType.MEMORY_EXHAUSTION:
            return f"Not enough memory to create your{chart_desc} chart. Your dataset might be too large for visualization."
        
        elif plotting_error_type == PlottingErrorType.TIMEOUT:
            return f"Creating your{chart_desc} chart is taking too long. The dataset might be too complex or large."
        
        elif plotting_error_type == PlottingErrorType.INVALID_CHART_TYPE:
            return f"The requested chart type '{chart_type}' is not supported or not suitable for your data."
        
        else:  # CHART_GENERATION
            return f"Unable to generate your{chart_desc} chart. There might be an issue with the data or chart configuration."
    
    def _get_plotting_recovery_suggestions(self, error_message: str,
                                         plotting_error_type: PlottingErrorType,
                                         chart_type: Optional[str] = None,
                                         data_columns: Optional[list[str]] = None,
                                         query: Optional[str] = None) -> list[str]:
        """
        Generate recovery suggestions for plotting errors.
        
        Args:
            error_message: The original error message
            plotting_error_type: The classified error type
            chart_type: The requested chart type
            data_columns: Available data columns
            query: Original user query
            
        Returns:
            list[str]: List of recovery suggestions
        """
        suggestions = []
        
        if plotting_error_type == PlottingErrorType.MATPLOTLIB_ERROR:
            suggestions.extend([
                "Try refreshing the page to reset the graphics system",
                "Ask for a simpler chart type like a bar chart or line plot",
                "Check if your browser supports chart rendering",
                "Try asking for the same chart with fewer data points"
            ])
        
        elif plotting_error_type == PlottingErrorType.PANDASAI_PLOTTING:
            suggestions.extend([
                "Try rephrasing your chart request more clearly",
                "Ask for a basic chart type like 'show me a bar chart'",
                "Specify the exact columns you want to visualize",
                "Try asking for a table view first, then request a chart"
            ])
        
        elif plotting_error_type == PlottingErrorType.DATA_COMPATIBILITY:
            suggestions.extend([
                "Check that your data has the right columns for this chart type",
                "Try a different chart type that matches your data better",
                "Use the data preview to see what columns are available",
                "Make sure numeric columns contain valid numbers"
            ])
            
            if data_columns:
                suggestions.append(f"Available columns: {', '.join(data_columns[:5])}")
        
        elif plotting_error_type == PlottingErrorType.MEMORY_EXHAUSTION:
            suggestions.extend([
                "Try filtering your data to include fewer rows",
                "Ask for a chart with a sample of your data",
                "Use aggregated data (like monthly totals instead of daily)",
                "Close other browser tabs to free up memory"
            ])
        
        elif plotting_error_type == PlottingErrorType.TIMEOUT:
            suggestions.extend([
                "Try a simpler chart with less data",
                "Filter your data to a smaller time range or subset",
                "Ask for a basic chart type first",
                "Check your internet connection"
            ])
        
        elif plotting_error_type == PlottingErrorType.INVALID_CHART_TYPE:
            suggestions.extend([
                "Try asking for a bar chart, line chart, or pie chart",
                "Use simpler chart names like 'plot', 'graph', or 'chart'",
                "Ask 'what charts can you make with this data?'",
                "Be more specific about what you want to visualize"
            ])
        
        else:  # CHART_GENERATION
            suggestions.extend([
                "Try asking for a different type of chart",
                "Make sure your data is loaded and visible",
                "Rephrase your request more clearly",
                "Ask for a simple bar chart or line plot first"
            ])
        
        # Add query-specific suggestions
        if query and len(query) > 100:
            suggestions.append("Try using a shorter, simpler question")
        
        return suggestions
    
    def _is_fallback_available(self, plotting_error_type: PlottingErrorType,
                             chart_type: Optional[str] = None) -> bool:
        """
        Determine if fallback chart generation is available for this error type.
        
        Args:
            plotting_error_type: The classified error type
            chart_type: The requested chart type
            
        Returns:
            bool: True if fallback is available
        """
        # Fallback available for most error types except memory exhaustion
        if plotting_error_type == PlottingErrorType.MEMORY_EXHAUSTION:
            return False
        
        # Fallback available for common chart types
        if chart_type and chart_type.lower() in ['bar', 'line', 'scatter', 'histogram', 'pie']:
            return True
        
        # General fallback available for most plotting errors
        return plotting_error_type in [
            PlottingErrorType.PANDASAI_PLOTTING,
            PlottingErrorType.MATPLOTLIB_ERROR,
            PlottingErrorType.CHART_GENERATION
        ]
    
    def _suggest_alternative_chart_type(self, plotting_error_type: PlottingErrorType,
                                      chart_type: Optional[str] = None,
                                      data_columns: Optional[list[str]] = None) -> Optional[str]:
        """
        Suggest alternative chart types based on error and data.
        
        Args:
            plotting_error_type: The classified error type
            chart_type: The requested chart type
            data_columns: Available data columns
            
        Returns:
            Optional[str]: Suggested alternative chart type
        """
        if plotting_error_type == PlottingErrorType.DATA_COMPATIBILITY:
            # Suggest simpler chart types for data compatibility issues
            if chart_type and chart_type.lower() in ['pie', 'donut']:
                return "bar chart"
            elif chart_type and chart_type.lower() in ['scatter', 'bubble']:
                return "line chart"
            else:
                return "bar chart"
        
        elif plotting_error_type == PlottingErrorType.MEMORY_EXHAUSTION:
            # Suggest simpler charts for memory issues
            return "bar chart with aggregated data"
        
        elif plotting_error_type == PlottingErrorType.INVALID_CHART_TYPE:
            # Suggest basic chart types
            return "bar chart"
        
        return None
    
    def _display_plotting_error_ui(self, plotting_error_info: PlottingErrorInfo):
        """
        Display plotting-specific error message in Streamlit UI.
        
        Args:
            plotting_error_info: The plotting error information
        """
        try:
            # Choose appropriate Streamlit component based on severity
            if plotting_error_info.severity == ErrorSeverity.CRITICAL:
                st.error(f"🚨 Critical Plotting Error: {plotting_error_info.user_message}")
            elif plotting_error_info.severity == ErrorSeverity.HIGH:
                st.error(f"📊❌ Chart Generation Error: {plotting_error_info.user_message}")
            elif plotting_error_info.severity == ErrorSeverity.MEDIUM:
                st.warning(f"📊⚠️ Chart Warning: {plotting_error_info.user_message}")
            else:
                st.info(f"📊ℹ️ Chart Notice: {plotting_error_info.user_message}")
            
            # Show plotting-specific information
            with st.expander("💡 Chart Generation Help"):
                if plotting_error_info.suggestions:
                    st.markdown("**Try these solutions:**")
                    for suggestion in plotting_error_info.suggestions:
                        st.markdown(f"• {suggestion}")
                
                if plotting_error_info.fallback_available:
                    st.success("✅ Automatic fallback chart generation is available")
                
                if plotting_error_info.suggested_chart_type:
                    st.info(f"💡 Suggested alternative: Try asking for a {plotting_error_info.suggested_chart_type}")
                
                if plotting_error_info.data_columns:
                    st.markdown("**Available data columns:**")
                    st.code(", ".join(plotting_error_info.data_columns[:10]))
                    
        except Exception as e:
            # Fallback if Streamlit is not available
            self.logger.warning(f"Could not display plotting error UI: {str(e)}")
            print(f"Plotting Error: {plotting_error_info.user_message}")
            if plotting_error_info.suggestions:
                print("Suggestions:")
                for suggestion in plotting_error_info.suggestions:
                    print(f"- {suggestion}")
    
    def _initialize_error_patterns(self) -> Dict[str, str]:
        """Initialize common error patterns and their user-friendly messages."""
        return {
            # File upload errors
            "file size": "The file is too large. Please use a file smaller than 50MB.",
            "unsupported format": "This file format is not supported. Please use CSV or Excel files.",
            "encoding": "There's an issue with the file encoding. Try saving your file with UTF-8 encoding.",
            "corrupted": "The file appears to be corrupted. Please check the file and try again.",
            "empty file": "The file appears to be empty. Please check your data and try again.",
            
            # Query processing errors
            "api key": "API key issue. Please check your GEMINI_API_KEY configuration.",
            "rate limit": "Too many requests. Please wait a moment and try again.",
            "timeout": "The query is taking too long. Please try a simpler question.",
            "invalid query": "The query format is not recognized. Please rephrase your question.",
            "no data": "No data is loaded. Please upload a dataset first.",
            
            # Visualization errors
            "matplotlib": "Error creating chart. Try asking for a different type of visualization.",
            "plotly": "Error creating interactive chart. Try a simpler visualization.",
            "no suitable data": "The data is not suitable for this type of visualization.",
            "too many points": "Too many data points to visualize. Try filtering your data first.",
            
            # Plotting-specific errors
            "no display name": "Chart display issue. The system cannot show charts in this environment.",
            "backend": "Chart rendering backend issue. Try refreshing the page.",
            "figure": "Problem creating the chart figure. Try a simpler chart type.",
            "axes": "Chart axis configuration error. Check your data columns.",
            "base64": "Chart encoding error. The generated chart couldn't be processed.",
            "smartdataframe": "PandasAI chart generation failed. Try rephrasing your request.",
            "generate_code": "AI code generation failed for chart. Try a simpler chart request.",
            "table name": "Data table reference error in chart generation.",
            "column not found": "Required column missing for chart. Check your data structure.",
            "insufficient data": "Not enough data points to create this chart type.",
            "empty dataframe": "No data available for chart generation.",
            "dtype": "Data type incompatible with requested chart type.",
            "string cannot be converted": "Text data cannot be used for numeric chart. Try a different chart type.",
            
            # System errors
            "out of memory": "Not enough memory to complete this operation.",
            "connection error": "Connection problem. Please check your internet connection.",
            "permission denied": "Permission denied. Please check file permissions.",
        }
    
    def _initialize_recovery_suggestions(self) -> Dict[ErrorCategory, list[str]]:
        """Initialize recovery suggestions for each error category."""
        return {
            ErrorCategory.FILE_UPLOAD: [
                "Ensure your file is in CSV or Excel format",
                "Check that the file size is under 50MB",
                "Verify the file is not corrupted",
                "Try saving the file with UTF-8 encoding"
            ],
            ErrorCategory.QUERY_PROCESSING: [
                "Try rephrasing your question more clearly",
                "Make sure you've uploaded data first",
                "Use simpler language in your query",
                "Check that column names in your query exist in the data"
            ],
            ErrorCategory.VISUALIZATION: [
                "Try asking for a different type of chart",
                "Ensure your data has the right format for visualization",
                "Consider filtering your data to reduce complexity",
                "Ask for a simpler visualization first"
            ],
            ErrorCategory.PLOTTING: [
                "Try asking for a basic chart type like bar chart or line plot",
                "Check that your data has the required columns for the chart",
                "Use simpler language when requesting charts",
                "Try filtering your data to fewer rows before plotting",
                "Ask for a table view first to verify your data structure",
                "Refresh the page if you're experiencing display issues"
            ],
            ErrorCategory.API_ERROR: [
                "Check your internet connection",
                "Verify your API key is configured correctly",
                "Wait a moment and try again",
                "Contact support if the problem persists"
            ],
            ErrorCategory.SESSION_MANAGEMENT: [
                "Try refreshing the page",
                "Clear your browser cache",
                "Start a new session",
                "Check if you have sufficient browser storage"
            ],
            ErrorCategory.SYSTEM_ERROR: [
                "Refresh the page and try again",
                "Check your internet connection",
                "Try using a different browser",
                "Contact support if the problem persists"
            ]
        }
    
    def _log_error(self, error_info: ErrorInfo):
        """Log error information appropriately based on severity."""
        log_message = f"[{error_info.category.value}] {error_info.message}"
        
        if error_info.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra={"error_info": error_info})
        elif error_info.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra={"error_info": error_info})
        elif error_info.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra={"error_info": error_info})
        else:
            self.logger.info(log_message, extra={"error_info": error_info})
        
        # Log technical details if available
        if error_info.technical_details:
            self.logger.debug(f"Technical details: {error_info.technical_details}")
    
    def _display_error_ui(self, error_info: ErrorInfo):
        """Display error message in Streamlit UI with appropriate styling."""
        try:
            # Choose appropriate Streamlit component based on severity
            if error_info.severity == ErrorSeverity.CRITICAL:
                st.error(f"🚨 Critical Error: {error_info.user_message}")
            elif error_info.severity == ErrorSeverity.HIGH:
                st.error(f"❌ Error: {error_info.user_message}")
            elif error_info.severity == ErrorSeverity.MEDIUM:
                st.warning(f"⚠️ Warning: {error_info.user_message}")
            else:
                st.info(f"ℹ️ Notice: {error_info.user_message}")
            
            # Show recovery suggestions if available
            if error_info.suggestions:
                with st.expander("💡 How to fix this"):
                    for suggestion in error_info.suggestions:
                        st.markdown(f"• {suggestion}")
        except Exception as e:
            # Fallback if Streamlit is not available (e.g., in tests)
            self.logger.warning(f"Could not display UI error message: {str(e)}")
            print(f"Error: {error_info.user_message}")  # Fallback to console
    
    def create_error_recovery_ui(self, error_info: ErrorInfo):
        """Create an interactive error recovery UI."""
        try:
            st.markdown("---")
            st.subheader("🔧 Error Recovery")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"**Error:** {error_info.user_message}")
                st.markdown(f"**Category:** {error_info.category.value.replace('_', ' ').title()}")
                st.markdown(f"**Time:** {error_info.timestamp.strftime('%H:%M:%S')}")
            
            with col2:
                if st.button("🔄 Try Again", type="primary"):
                    st.rerun()
                
                if st.button("🗑️ Clear Session", type="secondary"):
                    # Clear session and restart
                    for key in list(st.session_state.keys()):
                        del st.session_state[key]
                    st.rerun()
            
            # Show suggestions
            if error_info.suggestions:
                st.markdown("**Suggestions:**")
                for i, suggestion in enumerate(error_info.suggestions, 1):
                    st.markdown(f"{i}. {suggestion}")
        except Exception as e:
            # Fallback if Streamlit is not available
            self.logger.warning(f"Could not display recovery UI: {str(e)}")
            print(f"Error Recovery - {error_info.user_message}")
            if error_info.suggestions:
                print("Suggestions:")
                for suggestion in error_info.suggestions:
                    print(f"- {suggestion}")


# Global error handler instance
error_handler = ErrorHandler()


# Convenience functions for common error handling patterns
def handle_file_error(error: Union[Exception, str], filename: Optional[str] = None, show_ui: bool = True) -> ErrorInfo:
    """Handle file upload errors."""
    return error_handler.handle_file_upload_error(error, filename, show_ui)


def handle_query_error(error: Union[Exception, str], query: Optional[str] = None, show_ui: bool = True) -> ErrorInfo:
    """Handle query processing errors."""
    return error_handler.handle_query_processing_error(error, query, show_ui)


def handle_viz_error(error: Union[Exception, str], chart_type: Optional[str] = None, show_ui: bool = True) -> ErrorInfo:
    """Handle visualization errors."""
    return error_handler.handle_visualization_error(error, chart_type, show_ui)


def handle_plotting_error(error: Union[Exception, str], 
                         chart_type: Optional[str] = None,
                         data_columns: Optional[list[str]] = None,
                         query: Optional[str] = None,
                         show_ui: bool = True) -> PlottingErrorInfo:
    """Handle plotting-specific errors with enhanced classification."""
    return error_handler.handle_plotting_error(error, chart_type, data_columns, query, show_ui)


def safe_execute(func: Callable, error_category: ErrorCategory, 
                context: Optional[Dict[str, Any]] = None, 
                fallback_value: Any = None) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        error_category: Category of error if function fails
        context: Additional context for error handling
        fallback_value: Value to return if function fails
        
    Returns:
        Function result or fallback value
    """
    try:
        return func()
    except Exception as e:
        error_handler.handle_error(e, error_category, context)
        return fallback_value