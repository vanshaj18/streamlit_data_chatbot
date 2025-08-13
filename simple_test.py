#!/usr/bin/env python3
"""
Simple test to verify the main application components work together.
"""

import sys
import pandas as pd
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def test_basic_functionality():
    """Test basic functionality without mocking Streamlit."""
    print("🧪 Testing basic functionality...")
    
    # Test data structures
    from utils.session_manager import SessionState, ChatMessage, FileInfo
    
    # Test SessionState
    session = SessionState()
    assert session.dataframe is None
    assert len(session.chat_history) == 0
    assert session.is_initialized == False
    print("✅ SessionState creation works")
    
    # Test ChatMessage
    message = ChatMessage(
        role="user",
        content="Test message",
        timestamp=datetime.now(),
        message_type="text"
    )
    assert message.role == "user"
    assert message.content == "Test message"
    print("✅ ChatMessage creation works")
    
    # Test FileInfo
    file_info = FileInfo(
        filename="test.csv",
        file_size=1024,
        upload_timestamp=datetime.now(),
        columns=["col1", "col2"],
        row_count=100,
        file_type="csv"
    )
    assert file_info.filename == "test.csv"
    assert file_info.row_count == 100
    print("✅ FileInfo creation works")
    
    # Test DataFrame operations
    test_df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Carol'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })
    
    session.dataframe = test_df
    assert session.dataframe is not None
    assert len(session.dataframe) == 3
    print("✅ DataFrame storage works")
    
    # Test chat history
    session.chat_history.append(message)
    assert len(session.chat_history) == 1
    print("✅ Chat history works")


def test_file_validation():
    """Test file validation logic."""
    print("🧪 Testing file validation...")
    
    from components.file_handler import validate_file
    
    # Mock file object
    class MockFile:
        def __init__(self, name, size):
            self.name = name
            self.size = size
    
    # Test valid CSV
    valid_csv = MockFile("data.csv", 1024)
    is_valid, error = validate_file(valid_csv)
    assert is_valid == True
    print("✅ Valid CSV file passes validation")
    
    # Test valid Excel
    valid_excel = MockFile("data.xlsx", 2048)
    is_valid, error = validate_file(valid_excel)
    assert is_valid == True
    print("✅ Valid Excel file passes validation")
    
    # Test invalid format
    invalid_file = MockFile("data.txt", 1024)
    is_valid, error = validate_file(invalid_file)
    assert is_valid == False
    assert "Unsupported" in error
    print("✅ Invalid file format rejected")
    
    # Test file too large
    large_file = MockFile("data.csv", 100 * 1024 * 1024)  # 100MB
    is_valid, error = validate_file(large_file)
    assert is_valid == False
    assert "exceeds" in error
    print("✅ Large file rejected")


def test_agent_wrapper():
    """Test PandasAI agent wrapper."""
    print("🧪 Testing agent wrapper...")
    
    from services.pandas_agent import PandasAgent
    
    # Test agent creation
    agent = PandasAgent()
    assert agent is not None
    print("✅ Agent wrapper created successfully")
    
    # Test method existence (even if not implemented)
    assert hasattr(agent, 'initialize_agent')
    assert hasattr(agent, 'process_query')
    assert hasattr(agent, 'handle_response')
    print("✅ Agent methods exist")


def test_imports():
    """Test that all required modules can be imported."""
    print("🧪 Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported")
    except ImportError:
        print("⚠️  Streamlit not available (expected in test environment)")
    
    import pandas as pd
    print("✅ Pandas imported")
    
    import matplotlib.pyplot as plt
    print("✅ Matplotlib imported")
    
    import plotly.graph_objects as go
    print("✅ Plotly imported")
    
    # Test our modules
    from utils.session_manager import SessionState, ChatMessage, FileInfo
    from components.file_handler import validate_file
    from components.chat_interface import get_chat_stats
    from services.pandas_agent import PandasAgent
    print("✅ All custom modules imported")


def main():
    """Run all tests."""
    print("🚀 Starting Simple Integration Tests\n")
    
    try:
        test_imports()
        print()
        
        test_basic_functionality()
        print()
        
        test_file_validation()
        print()
        
        test_agent_wrapper()
        print()
        
        print("🎉 All tests passed!")
        print("\n📋 Verified Components:")
        print("✅ Data structures (SessionState, ChatMessage, FileInfo)")
        print("✅ File validation logic")
        print("✅ PandasAI agent wrapper")
        print("✅ Module imports and dependencies")
        print("\n🔄 Application is ready for:")
        print("  • File upload and data loading")
        print("  • Chat interface and message handling")
        print("  • Session state management")
        print("  • Integration with PandasAI (task 5)")
        print("  • Visualization rendering (task 6)")
        print("  • Error handling (task 7)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)