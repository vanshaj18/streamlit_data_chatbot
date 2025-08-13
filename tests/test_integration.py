#!/usr/bin/env python3
"""
Integration test for the Data Analysis Dashboard.
Tests the complete workflow: upload → query → response
"""

import sys
import os
import pandas as pd
from datetime import datetime
from unittest.mock import MagicMock

# Add current directory to path
sys.path.append('.')

def mock_streamlit():
    """Mock Streamlit components for testing."""
    import streamlit as st
    
    # Mock session state
    class MockSessionState:
        def __init__(self):
            self._state = {}
        
        def __contains__(self, key):
            return key in self._state
        
        def __getitem__(self, key):
            return self._state[key]
        
        def __setitem__(self, key, value):
            self._state[key] = value
        
        def __getattr__(self, key):
            if key.startswith('_'):
                raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")
            return self._state.get(key)
        
        def __setattr__(self, key, value):
            if key.startswith('_'):
                super().__setattr__(key, value)
            else:
                self._state[key] = value
        
        def get(self, key, default=None):
            return self._state.get(key, default)
        
        def __delitem__(self, key):
            if key in self._state:
                del self._state[key]
        
        def __delattr__(self, key):
            if key.startswith('_'):
                super().__delattr__(key)
            else:
                if key in self._state:
                    del self._state[key]
                else:
                    raise AttributeError(key)
    
    st.session_state = MockSessionState()
    
    # Mock other streamlit functions
    st.error = lambda x: print(f"ERROR: {x}")
    st.warning = lambda x: print(f"WARNING: {x}")
    st.info = lambda x: print(f"INFO: {x}")
    st.success = lambda x: print(f"SUCCESS: {x}")
    st.rerun = lambda: None


def test_session_management():
    """Test session state management."""
    print("🧪 Testing session management...")
    
    # Import after mocking to ensure proper session state handling
    import importlib
    import utils.session_manager
    importlib.reload(utils.session_manager)
    
    from utils.session_manager import (
        initialize_session, 
        get_session_data,
        update_dataframe,
        add_message,
        get_chat_history,
        has_dataframe,
        clear_session
    )
    
    # Test initialization
    initialize_session()
    session_data = get_session_data()
    assert session_data.is_initialized == True
    print("✅ Session initialization works")
    
    # Test DataFrame storage
    test_df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'age': [25, 30],
        'salary': [50000, 60000]
    })
    
    file_info = {
        'filename': 'test.csv',
        'file_size': 1024,
        'upload_timestamp': datetime.now(),
        'file_type': 'csv'
    }
    
    update_dataframe(test_df, file_info)
    
    update_dataframe(test_df, file_info)
    
    assert has_dataframe() == True
    print("✅ DataFrame storage works")
    
    # Test chat history
    add_message("user", "What is the average age?", "text")
    add_message("agent", "The average age is 27.5", "text")
    
    history = get_chat_history()
    assert len(history) == 2
    assert history[0].role == "user"
    assert history[1].role == "agent"
    print("✅ Chat history works")
    
    # Test session clearing
    clear_session()
    assert has_dataframe() == False
    assert len(get_chat_history()) == 0
    print("✅ Session clearing works")


def test_file_handling():
    """Test file upload and validation."""
    print("🧪 Testing file handling...")
    
    from components.file_handler import validate_file, load_dataframe
    
    # Mock uploaded file
    class MockUploadedFile:
        def __init__(self, name, size, content):
            self.name = name
            self.size = size
            self.content = content
            self._position = 0
        
        def seek(self, position):
            self._position = position
        
        def read(self):
            return self.content
    
    # Test CSV content
    csv_content = "name,age,salary\nAlice,25,50000\nBob,30,60000"
    
    # Test valid file
    valid_file = MockUploadedFile("test.csv", 1024, csv_content.encode())
    is_valid, error = validate_file(valid_file)
    assert is_valid == True
    print("✅ File validation works")
    
    # Test file too large
    large_file = MockUploadedFile("large.csv", 100 * 1024 * 1024, csv_content.encode())
    is_valid, error = validate_file(large_file)
    assert is_valid == False
    assert "exceeds" in error
    print("✅ File size validation works")
    
    # Test invalid format
    invalid_file = MockUploadedFile("test.txt", 1024, csv_content.encode())
    is_valid, error = validate_file(invalid_file)
    assert is_valid == False
    assert "Unsupported" in error
    print("✅ File format validation works")


def test_chat_interface():
    """Test chat interface components."""
    print("🧪 Testing chat interface...")
    
    from components.chat_interface import get_chat_stats, display_response
    from utils.session_manager import initialize_session, add_message
    
    # Initialize session
    initialize_session()
    
    # Test empty stats
    stats = get_chat_stats()
    assert stats["total_messages"] == 0
    print("✅ Empty chat stats work")
    
    # Add some messages
    add_message("user", "Hello", "text")
    add_message("agent", "Hi there!", "text")
    add_message("agent", "Error occurred", "error")
    
    # Test stats with messages
    stats = get_chat_stats()
    assert stats["total_messages"] == 3
    assert stats["user_messages"] == 1
    assert stats["agent_messages"] == 2
    assert stats["error_messages"] == 1
    print("✅ Chat statistics work")


def test_pandas_agent():
    """Test PandasAI agent wrapper."""
    print("🧪 Testing PandasAI agent...")
    
    from services.pandas_agent import PandasAgent
    
    # Test agent creation
    agent = PandasAgent()
    assert agent is not None
    print("✅ Agent creation works")
    
    # Note: Full agent functionality will be tested after task 5 implementation
    print("ℹ️  Full agent testing will be available after task 5")


def test_end_to_end_workflow():
    """Test the complete workflow."""
    print("🧪 Testing end-to-end workflow...")
    
    from utils.session_manager import (
        initialize_session, 
        update_dataframe, 
        has_dataframe,
        add_message,
        get_chat_history
    )
    
    # Step 1: Initialize session
    initialize_session()
    print("✅ Step 1: Session initialized")
    
    # Step 2: Upload data
    test_df = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'sales': [100, 200, 150],
        'profit': [20, 40, 30]
    })
    
    file_info = {
        'filename': 'sales_data.csv',
        'file_size': 2048,
        'upload_timestamp': datetime.now(),
        'file_type': 'csv'
    }
    
    update_dataframe(test_df, file_info)
    assert has_dataframe() == True
    print("✅ Step 2: Data uploaded and stored")
    
    # Step 3: Simulate user query
    user_query = "What is the total sales?"
    add_message("user", user_query, "text")
    print("✅ Step 3: User query added")
    
    # Step 4: Simulate agent response (placeholder until task 5)
    agent_response = "The total sales is 450 units."
    add_message("agent", agent_response, "text")
    print("✅ Step 4: Agent response added")
    
    # Step 5: Verify complete workflow
    history = get_chat_history()
    assert len(history) == 2
    assert history[0].content == user_query
    assert history[1].content == agent_response
    print("✅ Step 5: Complete workflow verified")


def main():
    """Run all integration tests."""
    print("🚀 Starting Data Analysis Dashboard Integration Tests\n")
    
    # Mock Streamlit
    mock_streamlit()
    
    try:
        # Run all tests
        test_session_management()
        print()
        
        test_file_handling()
        print()
        
        test_chat_interface()
        print()
        
        test_pandas_agent()
        print()
        
        test_end_to_end_workflow()
        print()
        
        print("🎉 All integration tests passed!")
        print("\n📋 Test Summary:")
        print("✅ Session state management")
        print("✅ File upload and validation")
        print("✅ Chat interface components")
        print("✅ PandasAI agent wrapper")
        print("✅ End-to-end workflow")
        print("\n🔄 Ready for tasks 5, 6, and 7 implementation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)