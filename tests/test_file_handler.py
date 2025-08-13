"""
Unit tests for file upload and data loading functionality.
"""

import pytest
import pandas as pd
import io
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from components.file_handler import (
    validate_file, 
    load_dataframe, 
    MAX_FILE_SIZE,
    SUPPORTED_TYPES
)


class TestValidateFile:
    """Test cases for file validation."""
    
    def test_validate_file_none(self):
        """Test validation with no file uploaded."""
        is_valid, error = validate_file(None)
        assert not is_valid
        assert "No file uploaded" in error
    
    def test_validate_file_too_large(self):
        """Test validation with file exceeding size limit."""
        mock_file = Mock()
        mock_file.size = MAX_FILE_SIZE + 1
        mock_file.name = "test.csv"
        
        is_valid, error = validate_file(mock_file)
        assert not is_valid
        assert "exceeds the 50MB limit" in error
    
    def test_validate_file_unsupported_format(self):
        """Test validation with unsupported file format."""
        mock_file = Mock()
        mock_file.size = 1000
        mock_file.name = "test.txt"
        
        is_valid, error = validate_file(mock_file)
        assert not is_valid
        assert "Unsupported file format" in error
    
    def test_validate_file_valid_csv(self):
        """Test validation with valid CSV file."""
        mock_file = Mock()
        mock_file.size = 1000
        mock_file.name = "test.csv"
        
        is_valid, error = validate_file(mock_file)
        assert is_valid
        assert error == ""
    
    def test_validate_file_valid_excel(self):
        """Test validation with valid Excel files."""
        for ext in ['xlsx', 'xls']:
            mock_file = Mock()
            mock_file.size = 1000
            mock_file.name = f"test.{ext}"
            
            is_valid, error = validate_file(mock_file)
            assert is_valid
            assert error == ""
    
    def test_validate_file_case_insensitive(self):
        """Test validation with uppercase file extensions."""
        mock_file = Mock()
        mock_file.size = 1000
        mock_file.name = "test.CSV"
        
        is_valid, error = validate_file(mock_file)
        assert is_valid
        assert error == ""


class TestLoadDataframe:
    """Test cases for DataFrame loading."""
    
    def create_mock_csv_file(self, content: str, filename: str = "test.csv"):
        """Helper to create mock CSV file."""
        mock_file = Mock()
        mock_file.name = filename
        mock_file.seek = Mock()
        mock_file.read = Mock(return_value=content.encode())
        
        # Create StringIO for pandas to read
        string_io = io.StringIO(content)
        
        # Mock the file to return StringIO when used as context manager
        mock_file.__enter__ = Mock(return_value=string_io)
        mock_file.__exit__ = Mock(return_value=None)
        
        return mock_file, string_io
    
    @patch('pandas.read_csv')
    def test_load_csv_success(self, mock_read_csv):
        """Test successful CSV loading."""
        # Create test DataFrame
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        mock_read_csv.return_value = test_df
        
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is not None
        assert error == ""
        assert len(df) == 3
        mock_read_csv.assert_called()
    
    @patch('pandas.read_excel')
    def test_load_excel_success(self, mock_read_excel):
        """Test successful Excel loading."""
        # Create test DataFrame
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': ['x', 'y', 'z']})
        mock_read_excel.return_value = test_df
        
        mock_file = Mock()
        mock_file.name = "test.xlsx"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is not None
        assert error == ""
        assert len(df) == 3
        mock_read_excel.assert_called()
    
    @patch('pandas.read_csv')
    def test_load_csv_encoding_fallback(self, mock_read_csv):
        """Test CSV loading with encoding fallback."""
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        
        # First call raises UnicodeDecodeError, second succeeds
        mock_read_csv.side_effect = [
            UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte'),
            test_df
        ]
        
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is not None
        assert error == ""
        assert mock_read_csv.call_count == 2
    
    @patch('pandas.read_csv')
    def test_load_csv_all_encodings_fail(self, mock_read_csv):
        """Test CSV loading when all encodings fail."""
        mock_read_csv.side_effect = UnicodeDecodeError('utf-8', b'', 0, 1, 'invalid start byte')
        
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is None
        assert "Unable to read CSV file" in error
    
    @patch('pandas.read_excel')
    def test_load_excel_error(self, mock_read_excel):
        """Test Excel loading with error."""
        mock_read_excel.side_effect = Exception("Excel read error")
        
        mock_file = Mock()
        mock_file.name = "test.xlsx"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is None
        assert "Unable to read Excel file" in error
    
    @patch('pandas.read_csv')
    def test_load_empty_dataframe(self, mock_read_csv):
        """Test loading empty DataFrame."""
        empty_df = pd.DataFrame()
        mock_read_csv.return_value = empty_df
        
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is None
        assert "appears to be empty" in error
    
    @patch('pandas.read_csv')
    def test_load_too_many_columns(self, mock_read_csv):
        """Test loading DataFrame with too many columns."""
        # Create DataFrame with 1001 columns
        columns = [f'col_{i}' for i in range(1001)]
        test_df = pd.DataFrame(columns=columns)
        test_df.loc[0] = [0] * 1001  # Add one row to avoid empty DataFrame
        mock_read_csv.return_value = test_df
        
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is None
        assert "too many columns" in error
    
    @patch('pandas.read_csv')
    def test_load_too_many_rows(self, mock_read_csv):
        """Test loading DataFrame with too many rows."""
        # Create DataFrame with 1M+ rows
        test_df = pd.DataFrame({'A': range(1000001)})
        mock_read_csv.return_value = test_df
        
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is None
        assert "too many rows" in error
    
    def test_load_unsupported_format(self):
        """Test loading unsupported file format."""
        mock_file = Mock()
        mock_file.name = "test.txt"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is None
        assert "Unsupported file format" in error
    
    @patch('pandas.read_csv')
    def test_load_general_exception(self, mock_read_csv):
        """Test handling of general exceptions during loading."""
        mock_read_csv.side_effect = Exception("General error")
        
        mock_file = Mock()
        mock_file.name = "test.csv"
        mock_file.seek = Mock()
        
        df, error = load_dataframe(mock_file)
        
        assert df is None
        assert "Error processing file" in error


class TestFileHandlerIntegration:
    """Integration tests for file handler functionality."""
    
    def test_supported_file_types_constant(self):
        """Test that supported file types constant is correct."""
        expected_types = ['csv', 'xlsx', 'xls']
        assert SUPPORTED_TYPES == expected_types
    
    def test_max_file_size_constant(self):
        """Test that max file size constant is 50MB."""
        expected_size = 50 * 1024 * 1024  # 50MB in bytes
        assert MAX_FILE_SIZE == expected_size
    
    @patch('components.file_handler.update_dataframe')
    @patch('pandas.read_csv')
    def test_end_to_end_csv_processing(self, mock_read_csv, mock_update_dataframe):
        """Test end-to-end CSV file processing."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'Name': ['Alice', 'Bob', 'Charlie'],
            'Age': [25, 30, 35],
            'City': ['New York', 'London', 'Tokyo']
        })
        mock_read_csv.return_value = test_df
        
        # Create mock file
        mock_file = Mock()
        mock_file.name = "test_data.csv"
        mock_file.size = 1000
        mock_file.seek = Mock()
        
        # Test validation
        is_valid, error = validate_file(mock_file)
        assert is_valid
        assert error == ""
        
        # Test loading
        df, load_error = load_dataframe(mock_file)
        assert df is not None
        assert load_error == ""
        assert len(df) == 3
        assert list(df.columns) == ['Name', 'Age', 'City']
    
    @patch('components.file_handler.update_dataframe')
    @patch('pandas.read_excel')
    def test_end_to_end_excel_processing(self, mock_read_excel, mock_update_dataframe):
        """Test end-to-end Excel file processing."""
        # Create test DataFrame
        test_df = pd.DataFrame({
            'Product': ['A', 'B', 'C'],
            'Sales': [100, 200, 150],
            'Profit': [20, 40, 30]
        })
        mock_read_excel.return_value = test_df
        
        # Create mock file
        mock_file = Mock()
        mock_file.name = "sales_data.xlsx"
        mock_file.size = 2000
        mock_file.seek = Mock()
        
        # Test validation
        is_valid, error = validate_file(mock_file)
        assert is_valid
        assert error == ""
        
        # Test loading
        df, load_error = load_dataframe(mock_file)
        assert df is not None
        assert load_error == ""
        assert len(df) == 3
        assert list(df.columns) == ['Product', 'Sales', 'Profit']


if __name__ == "__main__":
    pytest.main([__file__])