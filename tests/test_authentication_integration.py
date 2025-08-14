#!/usr/bin/env python3
"""
Authentication integration tests for the Data Chatbot Dashboard.
Tests that the Gemini API authentication is working correctly.
"""

import sys
import os
import pandas as pd
from unittest.mock import patch, Mock
from datetime import datetime

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
        
        def get(self, key, default=None):
            return self._state.get(key, default)
    
    st.session_state = MockSessionState()
    st.error = lambda x: print(f"ERROR: {x}")
    st.success = lambda x: print(f"SUCCESS: {x}")
    st.rerun = lambda: None


def test_api_key_configuration():
    """Test that API key is properly configured."""
    print("🧪 Testing API key configuration...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key is not None, "GEMINI_API_KEY must be set in environment"
    assert len(api_key) > 20, "GEMINI_API_KEY appears to be invalid (too short)"
    assert api_key.startswith("AIza"), "GEMINI_API_KEY should start with 'AIza'"
    
    print("✅ API key is properly configured")


def test_pandas_agent_initialization():
    """Test that PandasAI agent initializes correctly with API key."""
    print("🧪 Testing PandasAI agent initialization...")
    
    mock_streamlit()
    
    from services.pandas_agent import PandasAgent
    
    # Test agent creation with API key
    agent = PandasAgent()
    assert agent.api_key is not None, "Agent should have API key"
    
    # Test with sample data
    test_df = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie'],
        'age': [25, 30, 35],
        'salary': [50000, 60000, 70000]
    })
    
    # Initialize agent with DataFrame
    success = agent.initialize_agent(test_df)
    assert success == True, "Agent initialization should succeed with valid API key"
    assert agent.is_initialized() == True, "Agent should be properly initialized"
    
    print("✅ PandasAI agent initialization successful")


def test_real_api_connection():
    """Test actual connection to Gemini API with a simple query."""
    print("🧪 Testing real API connection...")
    
    mock_streamlit()
    
    from services.pandas_agent import PandasAgent
    
    # Create agent
    agent = PandasAgent()
    
    # Test data
    test_df = pd.DataFrame({
        'product': ['A', 'B', 'C'],
        'sales': [100, 200, 150],
        'price': [10, 20, 15]
    })
    
    # Initialize agent
    success = agent.initialize_agent(test_df)
    assert success == True, "Agent should initialize successfully"
    
    # Test a simple query that should work
    simple_query = "What is the total sales?"
    response = agent.process_query(simple_query)
    
    # Verify response
    assert response is not None, "Response should not be None"
    assert response.response_type != "error", f"Query should not fail. Error: {response.error_message}"
    assert response.execution_time > 0, "Execution time should be positive"
    
    print(f"✅ Real API connection successful. Response type: {response.response_type}")
    print(f"   Query: {simple_query}")
    print(f"   Response: {response.content}")


def test_authentication_error_handling():
    """Test that authentication errors are handled gracefully."""
    print("🧪 Testing authentication error handling...")
    
    mock_streamlit()
    
    from services.pandas_agent import PandasAgent
    
    # Test with invalid API key
    agent = PandasAgent(api_key="invalid_key")
    
    test_df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'value': [1, 2]
    })
    
    # This should fail gracefully
    success = agent.initialize_agent(test_df)
    
    if not success:
        print("✅ Invalid API key properly rejected during initialization")
    else:
        # If initialization succeeds (shouldn't happen), test query failure
        response = agent.process_query("What is the sum?")
        assert response.response_type == "error", "Invalid API key should result in error response"
        print("✅ Invalid API key properly handled during query processing")


def test_multiple_queries_session():
    """Test multiple queries in a single session to ensure connection stability."""
    print("🧪 Testing multiple queries in session...")
    
    mock_streamlit()
    
    from services.pandas_agent import PandasAgent
    
    # Create agent
    agent = PandasAgent()
    
    # Test data
    test_df = pd.DataFrame({
        'department': ['Sales', 'Engineering', 'Marketing', 'Sales', 'Engineering'],
        'employee': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'salary': [50000, 80000, 60000, 55000, 85000],
        'years': [2, 5, 3, 1, 6]
    })
    
    # Initialize agent
    success = agent.initialize_agent(test_df)
    assert success == True, "Agent should initialize successfully"
    
    # Test multiple queries
    queries = [
        "How many employees are there?",
        "What is the average salary?",
        "Which department has the highest average salary?"
    ]
    
    successful_queries = 0
    
    for i, query in enumerate(queries):
        print(f"   Query {i+1}: {query}")
        response = agent.process_query(query)
        
        if response.response_type != "error":
            successful_queries += 1
            print(f"   ✅ Success: {response.content}")
        else:
            print(f"   ⚠️ Error: {response.error_message}")
    
    # At least 2 out of 3 queries should succeed
    assert successful_queries >= 2, f"At least 2 queries should succeed, got {successful_queries}"
    
    print(f"✅ Multiple queries test completed. {successful_queries}/{len(queries)} queries successful")


def test_data_types_and_formats():
    """Test that the agent handles different data types correctly."""
    print("🧪 Testing data types and formats...")
    
    mock_streamlit()
    
    from services.pandas_agent import PandasAgent
    
    # Create agent
    agent = PandasAgent()
    
    # Test data with various types
    test_df = pd.DataFrame({
        'id': [1, 2, 3, 4, 5],
        'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
        'salary': [50000.0, 60000.5, 70000.0, 80000.25, 90000.0],
        'is_manager': [False, True, False, True, False],
        'start_date': pd.to_datetime(['2020-01-01', '2019-06-15', '2021-03-10', '2018-11-20', '2022-02-28'])
    })
    
    # Initialize agent
    success = agent.initialize_agent(test_df)
    assert success == True, "Agent should handle mixed data types"
    
    # Test a query that involves different data types
    response = agent.process_query("How many managers are there?")
    
    assert response.response_type != "error", f"Mixed data types query failed: {response.error_message}"
    
    print("✅ Mixed data types handled successfully")


def run_authentication_tests():
    """Run all authentication integration tests."""
    print("🚀 Starting Authentication Integration Tests\n")
    
    try:
        test_api_key_configuration()
        print()
        
        test_pandas_agent_initialization()
        print()
        
        test_real_api_connection()
        print()
        
        test_authentication_error_handling()
        print()
        
        test_multiple_queries_session()
        print()
        
        test_data_types_and_formats()
        print()
        
        print("🎉 All authentication integration tests passed!")
        print("\n📋 Authentication Test Summary:")
        print("✅ API key configuration")
        print("✅ PandasAI agent initialization")
        print("✅ Real API connection")
        print("✅ Authentication error handling")
        print("✅ Multiple queries session stability")
        print("✅ Mixed data types support")
        print("\n🔐 Authentication Status: WORKING")
        print("🌐 Gemini API Connection: SUCCESSFUL")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Authentication integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_authentication_tests()
    sys.exit(0 if success else 1)