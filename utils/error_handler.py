"""
Centralized error handling system for the data analysis dashboard.
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
            "api key": "API key issue. Please check your GooglePalm API key configuration.",
            "rate limit": "Too many requests. Please wait a moment and try again.",
            "timeout": "The query is taking too long. Please try a simpler question.",
            "invalid query": "The query format is not recognized. Please rephrase your question.",
            "no data": "No data is loaded. Please upload a dataset first.",
            
            # Visualization errors
            "matplotlib": "Error creating chart. Try asking for a different type of visualization.",
            "plotly": "Error creating interactive chart. Try a simpler visualization.",
            "no suitable data": "The data is not suitable for this type of visualization.",
            "too many points": "Too many data points to visualize. Try filtering your data first.",
            
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