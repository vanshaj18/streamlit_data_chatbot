"""
File upload and data loading components.
"""

import streamlit as st
import pandas as pd
import io
from typing import Optional, Tuple, Dict, Any
from datetime import datetime
from utils.session_manager import update_dataframe, get_file_info
from utils.session_manager import clear_session
from utils.error_handler import handle_file_error, safe_execute, ErrorCategory


# File size limit in bytes (50MB)
MAX_FILE_SIZE = 50 * 1024 * 1024

# Supported file types
SUPPORTED_TYPES = ['csv', 'xlsx', 'xls']


def validate_file(uploaded_file) -> Tuple[bool, str]:
    """
    Validate uploaded file format and size.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple[bool, str]: (is_valid, error_message)
    """
    try:
        if uploaded_file is None:
            error_info = handle_file_error("No file uploaded", None, show_ui=False)
            return False, error_info.user_message
        
        # Check file size
        if uploaded_file.size > MAX_FILE_SIZE:
            size_mb = uploaded_file.size / (1024 * 1024)
            error_msg = f"File size ({size_mb:.1f}MB) exceeds the 50MB limit"
            error_info = handle_file_error(error_msg, uploaded_file.name, show_ui=False)
            return False, error_info.user_message
        
        # Check file extension
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in SUPPORTED_TYPES:
            error_msg = f"Unsupported file format: {file_extension}"
            error_info = handle_file_error(error_msg, uploaded_file.name, show_ui=False)
            return False, error_info.user_message
        
        return True, ""
        
    except Exception as e:
        error_info = handle_file_error(e, uploaded_file.name if uploaded_file else None, show_ui=False)
        return False, error_info.user_message


def load_dataframe(uploaded_file) -> Tuple[Optional[pd.DataFrame], str]:
    """
    Convert uploaded file to pandas DataFrame.
    
    Args:
        uploaded_file: Streamlit UploadedFile object
        
    Returns:
        Tuple[Optional[pd.DataFrame], str]: (dataframe, error_message)
    """
    def _load_dataframe_internal():
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        # Reset file pointer to beginning
        uploaded_file.seek(0)
        
        if file_extension == 'csv':
            # Try different encodings for CSV files
            encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    df = pd.read_csv(uploaded_file, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            else:
                raise ValueError("Unable to read CSV file with any supported encoding")
                
        elif file_extension in ['xlsx', 'xls']:
            df = pd.read_excel(uploaded_file)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
        
        # Validate DataFrame
        if df.empty:
            raise ValueError("The uploaded file appears to be empty")
        
        # Check for reasonable number of columns (prevent memory issues)
        if len(df.columns) > 1000:
            raise ValueError("File has too many columns (>1000). Please use a smaller dataset.")
        
        # Check for reasonable number of rows
        if len(df) > 1000000:
            raise ValueError("File has too many rows (>1M). Please use a smaller dataset.")
        
        return df
    
    try:
        df = _load_dataframe_internal()
        return df, ""
    except Exception as e:
        error_info = handle_file_error(e, uploaded_file.name, show_ui=False)
        return None, error_info.user_message


def display_data_preview(df: pd.DataFrame, file_info: Dict[str, Any]) -> None:
    """
    Display a preview of the uploaded data.
    
    Args:
        df: The pandas DataFrame
        file_info: Dictionary containing file metadata
    """
    st.success("✅ File uploaded successfully!")
    
    # Display file information
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Rows", f"{len(df):,}")
    
    with col2:
        st.metric("Columns", len(df.columns))
    
    with col3:
        size_mb = file_info['file_size'] / (1024 * 1024)
        st.metric("Size", f"{size_mb:.1f} MB")
    
    # Display data types
    with st.expander("📋 Column Information"):
        col_info = pd.DataFrame({
            'Column': df.columns,
            'Data Type': df.dtypes.astype(str),
            'Non-Null Count': df.count(),
            'Null Count': df.isnull().sum()
        })
        st.dataframe(col_info, use_container_width=True)
    
    # Display data preview
    with st.expander("👀 Data Preview (First 10 rows)", expanded=True):
        st.dataframe(df.head(5), use_container_width=True)      
    
    # Display basic statistics for numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    if len(numeric_cols) > 0:
        with st.expander("📊 Basic Statistics"):
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)


def render_file_upload():
    """Render the file upload component."""
    st.markdown("Upload a CSV or Excel file to get started with your data analysis.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose a file",
        type=SUPPORTED_TYPES,
        help=f"Supported formats: {', '.join(SUPPORTED_TYPES).upper()}. Maximum size: 30MB"
    )
    
    if uploaded_file is not None:
        # Validate file
        is_valid, error_message = validate_file(uploaded_file)
        
        if not is_valid:
            # Error message is already user-friendly from validation
            st.error(f"❌ {error_message}")
            return
        
        # Show loading spinner while processing
        with st.spinner("Processing file..."):
            # Load DataFrame
            df, load_error = load_dataframe(uploaded_file)
            
            if load_error:
                # Error message is already user-friendly from load_dataframe
                st.error(f"❌ {load_error}")
                return
            
            # Create file info dictionary
            file_info = {
                'filename': uploaded_file.name,
                'file_size': uploaded_file.size,
                'upload_timestamp': datetime.now(),
                'file_type': uploaded_file.name.split('.')[-1].lower()
            }
            
            # Update session state
            update_dataframe(df, file_info)
            
            # Display preview
            display_data_preview(df, file_info)
    
    else:
        # Show current file info if one is already loaded
        current_file_info = get_file_info()
        if current_file_info:
            st.info(f"📁 Currently loaded: **{current_file_info.filename}** "
                   f"({current_file_info.row_count:,} rows, {len(current_file_info.columns)} columns)")
            
            # Option to clear current data
            if st.button("🗑️ Clear Current Data", type="secondary"):
                clear_session()
                st.rerun()
        else:
            st.info("👆 Upload a file to begin analyzing your data!")