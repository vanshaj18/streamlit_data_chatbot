"""
Integration tests for file handler with real file operations.
"""

import pytest
import pandas as pd
import tempfile
import os
from components.file_handler import validate_file, load_dataframe
from unittest.mock import Mock


class MockUploadedFile:
    """Mock Streamlit UploadedFile for testing."""
    
    def __init__(self, filename, content, file_type='csv'):
        self.name = filename
        self.content = content
        self.size = len(content.encode()) if isinstance(content, str) else len(content)
        self.file_type = file_type
        self._position = 0
    
    def seek(self, position):
        self._position = position
    
    def read(self):
        return self.content.encode() if isinstance(self.content, str) else self.content
    
    def getvalue(self):
        return self.content.encode() if isinstance(self.content, str) else self.content


class TestRealFileOperations:
    """Test file operations with real data."""
    
    def test_load_real_csv_file(self):
        """Test loading a real CSV file."""
        csv_content = """Name,Age,City
Alice,25,New York
Bob,30,London
Charlie,35,Tokyo"""
        
        mock_file = MockUploadedFile("test.csv", csv_content)
        
        # Test validation
        is_valid, error = validate_file(mock_file)
        assert is_valid
        assert error == ""
        
        # Mock pandas.read_csv to use our content
        import pandas as pd
        from unittest.mock import patch
        import io
        
        with patch('pandas.read_csv') as mock_read_csv:
            expected_df = pd.DataFrame({
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', 'London', 'Tokyo']
            })
            mock_read_csv.return_value = expected_df
            
            df, load_error = load_dataframe(mock_file)
            
            assert df is not None
            assert load_error == ""
            assert len(df) == 3
            assert list(df.columns) == ['Name', 'Age', 'City']
    
    def test_load_csv_with_special_characters(self):
        """Test loading CSV with special characters."""
        csv_content = """Name,Description
José,Café owner
François,Résumé writer
München,Größe measurement"""
        
        mock_file = MockUploadedFile("special.csv", csv_content)
        
        # Test validation
        is_valid, error = validate_file(mock_file)
        assert is_valid
        
        # Mock pandas to handle encoding
        import pandas as pd
        from unittest.mock import patch
        
        with patch('pandas.read_csv') as mock_read_csv:
            expected_df = pd.DataFrame({
                'Name': ['José', 'François', 'München'],
                'Description': ['Café owner', 'Résumé writer', 'Größe measurement']
            })
            mock_read_csv.return_value = expected_df
            
            df, load_error = load_dataframe(mock_file)
            
            assert df is not None
            assert load_error == ""
    
    def test_load_large_file_simulation(self):
        """Test handling of large file (simulated)."""
        # Create a mock file that exceeds size limit
        large_mock_file = Mock()
        large_mock_file.name = "large.csv"
        large_mock_file.size = 60 * 1024 * 1024  # 60MB
        
        is_valid, error = validate_file(large_mock_file)
        assert not is_valid
        assert "exceeds the 50MB limit" in error
    
    def test_load_empty_csv(self):
        """Test loading empty CSV file."""
        csv_content = ""
        mock_file = MockUploadedFile("empty.csv", csv_content)
        
        import pandas as pd
        from unittest.mock import patch
        
        with patch('pandas.read_csv') as mock_read_csv:
            mock_read_csv.return_value = pd.DataFrame()  # Empty DataFrame
            
            df, load_error = load_dataframe(mock_file)
            
            assert df is None
            assert "appears to be empty" in load_error
    
    def test_load_csv_with_only_headers(self):
        """Test loading CSV with only headers."""
        csv_content = "Name,Age,City"
        mock_file = MockUploadedFile("headers_only.csv", csv_content)
        
        import pandas as pd
        from unittest.mock import patch
        
        with patch('pandas.read_csv') as mock_read_csv:
            # DataFrame with columns but no data rows
            mock_read_csv.return_value = pd.DataFrame(columns=['Name', 'Age', 'City'])
            
            df, load_error = load_dataframe(mock_file)
            
            assert df is None
            assert "appears to be empty" in load_error
    
    def test_load_malformed_csv(self):
        """Test loading malformed CSV file."""
        csv_content = """Name,Age,City
Alice,25,New York,Extra
Bob,30
Charlie,35,Tokyo,Another,Extra"""
        
        mock_file = MockUploadedFile("malformed.csv", csv_content)
        
        import pandas as pd
        from unittest.mock import patch
        
        with patch('pandas.read_csv') as mock_read_csv:
            # Pandas usually handles malformed CSV gracefully
            expected_df = pd.DataFrame({
                'Name': ['Alice', 'Bob', 'Charlie'],
                'Age': [25, 30, 35],
                'City': ['New York', None, 'Tokyo']
            })
            mock_read_csv.return_value = expected_df
            
            df, load_error = load_dataframe(mock_file)
            
            assert df is not None
            assert load_error == ""
    
    def test_file_extension_case_insensitive(self):
        """Test that file extensions are handled case-insensitively."""
        extensions = ['CSV', 'csv', 'Csv', 'XLSX', 'xlsx', 'XLS', 'xls']
        
        for ext in extensions:
            mock_file = Mock()
            mock_file.name = f"test.{ext}"
            mock_file.size = 1000
            
            is_valid, error = validate_file(mock_file)
            assert is_valid, f"Extension {ext} should be valid"
            assert error == ""


if __name__ == "__main__":
    pytest.main([__file__])