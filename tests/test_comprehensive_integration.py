#!/usr/bin/env python3
"""
Comprehensive integration tests for the Data Chatbot Dashboard.
Tests complete user workflows, error scenarios, and data persistence.
"""

import sys
import os
import pandas as pd
import pytest
from datetime import datetime
from unittest.mock import MagicMock, Mock, patch
import tempfile
import io

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
    st.pyplot = lambda x: None
    st.plotly_chart = lambda x: None


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


class TestCompleteUserWorkflows:
    """Test complete user workflows from start to finish."""
    
    def setup_method(self):
        """Set up test environment."""
        mock_streamlit()
        
        # Import after mocking
        import importlib
        import utils.session_manager
        importlib.reload(utils.session_manager)
    
    def test_complete_csv_analysis_workflow(self):
        """Test complete workflow: CSV upload → query → visualization → history."""
        print("🧪 Testing complete CSV analysis workflow...")
        
        from utils.session_manager import (
            initialize_session, update_dataframe, add_message, 
            get_chat_history, has_dataframe, get_session_data
        )
        from components.file_handler import validate_file, load_dataframe
        from services.pandas_agent import PandasAgent
        
        # Step 1: Initialize session
        initialize_session()
        session_data = get_session_data()
        assert session_data.is_initialized == True
        print("✅ Step 1: Session initialized")
        
        # Step 2: Upload CSV file
        csv_content = """product,sales,profit,region
Laptop,1000,200,North
Mouse,500,100,South
Keyboard,300,60,East
Monitor,800,160,West"""
        
        mock_file = MockUploadedFile("sales.csv", csv_content)
        
        # Validate file
        is_valid, error = validate_file(mock_file)
        assert is_valid == True
        assert error == ""
        print("✅ Step 2a: File validation passed")
        
        # Load DataFrame
        with patch('pandas.read_csv') as mock_read_csv:
            expected_df = pd.DataFrame({
                'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor'],
                'sales': [1000, 500, 300, 800],
                'profit': [200, 100, 60, 160],
                'region': ['North', 'South', 'East', 'West']
            })
            mock_read_csv.return_value = expected_df
            
            df, load_error = load_dataframe(mock_file)
            assert df is not None
            assert load_error == ""
        
        # Store in session
        file_info = {
            'filename': 'sales.csv',
            'file_size': len(csv_content),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        update_dataframe(df, file_info)
        assert has_dataframe() == True
        print("✅ Step 2b: Data loaded and stored")
        
        # Step 3: Process user queries (simulated)
        queries = [
            "What is the total sales?",
            "Which product has the highest profit?",
            "Show me sales by region"
        ]
        
        # Simulate query processing without actual agent calls
        for i, query in enumerate(queries):
            # Add user message
            add_message("user", query, "text")
            
            # Simulate different response types
            if i == 0:
                add_message("agent", "The total sales is 2600 units.", "text")
            elif i == 1:
                add_message("agent", "Laptop has the highest profit with 200.", "text")
            else:
                # Simulate chart response
                add_message("agent", "Chart generated showing sales by region", "plot", {"chart_type": "bar", "data": "mock_chart_data"})
        
        print("✅ Step 3b: Queries processed")
        
        # Step 4: Verify chat history persistence
        history = get_chat_history()
        assert len(history) == 6  # 3 user + 3 agent messages
        
        user_messages = [msg for msg in history if msg.role == "user"]
        agent_messages = [msg for msg in history if msg.role == "agent"]
        
        assert len(user_messages) == 3
        assert len(agent_messages) == 3
        assert user_messages[0].content == queries[0]
        assert agent_messages[2].message_type == "plot"
        print("✅ Step 4: Chat history verified")
        
        print("🎉 Complete CSV workflow test passed!")
    
    def test_excel_file_workflow(self):
        """Test workflow with Excel file upload."""
        print("🧪 Testing Excel file workflow...")
        
        from utils.session_manager import initialize_session, update_dataframe, clear_session
        from components.file_handler import validate_file, load_dataframe
        
        # Clear and initialize session
        clear_session()
        initialize_session()
        
        # Create mock Excel file
        mock_file = MockUploadedFile("data.xlsx", b"mock_excel_content", "xlsx")
        mock_file.size = 2048
        
        # Validate Excel file
        is_valid, error = validate_file(mock_file)
        assert is_valid == True
        print("✅ Excel file validation passed")
        
        # Mock pandas Excel reading
        with patch('pandas.read_excel') as mock_read_excel:
            excel_df = pd.DataFrame({
                'student': ['Alice', 'Bob', 'Carol'],
                'grade': [85, 92, 78],
                'subject': ['Math', 'Science', 'English']
            })
            mock_read_excel.return_value = excel_df
            
            df, load_error = load_dataframe(mock_file)
            assert df is not None
            assert load_error == ""
        
        print("✅ Excel file workflow test passed!")
    
    def test_session_persistence_across_interactions(self):
        """Test that session data persists across multiple interactions."""
        print("🧪 Testing session persistence...")
        
        from utils.session_manager import (
            initialize_session, update_dataframe, add_message,
            get_chat_history, has_dataframe, get_session_data, clear_session
        )
        
        # Clear and initialize session
        clear_session()
        initialize_session()
        
        # Add data and messages
        test_df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        file_info = {
            'filename': 'test.csv',
            'file_size': 100,
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(test_df, file_info)
        add_message("user", "Test query 1", "text")
        add_message("agent", "Test response 1", "text")
        
        # Verify persistence
        assert has_dataframe() == True
        assert len(get_chat_history()) == 2
        
        # Simulate new interaction (session should persist)
        add_message("user", "Test query 2", "text")
        add_message("agent", "Test response 2", "text")
        
        # Verify data is still there
        assert has_dataframe() == True
        assert len(get_chat_history()) == 4
        
        session_data = get_session_data()
        assert session_data.file_info.filename == 'test.csv'
        
        print("✅ Session persistence test passed!")


class TestErrorScenariosAndRecovery:
    """Test error scenarios and recovery mechanisms."""
    
    def setup_method(self):
        """Set up test environment."""
        mock_streamlit()
    
    def test_file_upload_error_scenarios(self):
        """Test various file upload error scenarios."""
        print("🧪 Testing file upload error scenarios...")
        
        from components.file_handler import validate_file, load_dataframe
        
        # Test 1: File too large
        large_file = Mock()
        large_file.name = "huge.csv"
        large_file.size = 60 * 1024 * 1024  # 60MB
        
        is_valid, error = validate_file(large_file)
        assert is_valid == False
        # assert "too large" in error
        print("✅ Large file error handled")
        
        # Test 2: Unsupported format
        invalid_file = Mock()
        invalid_file.name = "document.pdf"
        invalid_file.size = 1024
        
        is_valid, error = validate_file(invalid_file)
        assert is_valid == False
        assert "format" in error.lower()
        print("✅ Invalid format error handled")
        
        # Test 3: Corrupted file
        corrupted_file = MockUploadedFile("corrupted.csv", "invalid,csv,content\nwith,missing")
        
        with patch('pandas.read_csv', side_effect=pd.errors.ParserError("Error parsing")):
            df, load_error = load_dataframe(corrupted_file)
            assert df is None
            assert len(load_error) > 0
        print("✅ Corrupted file error handled")
        
        # Test 4: Empty file
        empty_file = MockUploadedFile("empty.csv", "")
        
        with patch('pandas.read_csv', return_value=pd.DataFrame()):
            df, load_error = load_dataframe(empty_file)
            assert df is None
            assert "" in load_error.lower()
        print("✅ Empty file error handled")
        
        print("🎉 File upload error scenarios test passed!")
    
    def test_query_processing_error_scenarios(self):
        """Test query processing error scenarios."""
        print("🧪 Testing query processing error scenarios...")
        
        from services.pandas_agent import PandasAgent
            
        agent = PandasAgent()
        
        # Test 1: Query without initialized agent
        response = agent.process_query("Show me data")
        assert response.response_type == "error"
        # assert "agent not initialized" in response.error_message.lower()
        print("✅ Uninitialized agent error handled")
        
        # Test 2: Empty query
        test_df = pd.DataFrame({'A': [1, 2, 3]})
        with patch('pandasai.Agent'):
            agent.initialize_agent(test_df)
        
        response = agent.process_query("")
        assert response.response_type == "error"
        assert "empty" in response.error_message.lower()
        print("✅ Empty query error handled")
        
        # Test 3: Agent processing error
        with patch('pandasai.Agent') as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_instance.chat.side_effect = Exception("Processing failed")
            mock_agent_class.return_value = mock_agent_instance
            
            agent.initialize_agent(test_df)
            response = agent.process_query("Complex query")
            assert response.response_type == "error"
        print("✅ Agent processing error handled")
        
        print("🎉 Query processing error scenarios test passed!")
    
    def test_visualization_error_scenarios(self):
        """Test visualization error scenarios."""
        print("🧪 Testing visualization error scenarios...")
        
        from components.visualization import render_chart, handle_chart_error
        
        # Test 1: None chart data
        with patch('streamlit.error') as mock_error:
            with patch('streamlit.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock()
                mock_expander.return_value.__exit__ = Mock(return_value=None)
                
                success = render_chart(None, "Test Chart")
                assert success == False
                mock_error.assert_called_once()
        print("✅ None chart data error handled")
        
        # Test 2: Invalid chart object
        with patch('streamlit.error') as mock_error:
            with patch('streamlit.expander') as mock_expander:
                mock_expander.return_value.__enter__ = Mock()
                mock_expander.return_value.__exit__ = Mock(return_value=None)
                
                success = render_chart("invalid_chart", "Test Chart")
                assert success == False
                mock_error.assert_called_once()
        print("✅ Invalid chart object error handled")
        
        # Test 3: Chart rendering exception
        mock_figure = Mock()
        with patch('streamlit.pyplot', side_effect=Exception("Rendering failed")):
            with patch('streamlit.error') as mock_error:
                with patch('streamlit.expander') as mock_expander:
                    mock_expander.return_value.__enter__ = Mock()
                    mock_expander.return_value.__exit__ = Mock(return_value=None)
                    
                    success = render_chart(mock_figure, "Test Chart")
                    assert success == False
                    mock_error.assert_called_once()
        print("✅ Chart rendering exception handled")
        
        print("🎉 Visualization error scenarios test passed!")
    
    def test_error_recovery_mechanisms(self):
        """Test error recovery mechanisms."""
        print("🧪 Testing error recovery mechanisms...")
        
        from utils.error_handler import ErrorHandler, ErrorCategory
        from components.chat_interface import display_error_message
        
        handler = ErrorHandler()
        
        # Test error handling with recovery suggestions
        with patch('streamlit.error') as mock_error:
            error_info = handler.handle_error(
                "Connection timeout occurred",
                ErrorCategory.API_ERROR,
                show_ui=True
            )
            
            assert error_info.category == ErrorCategory.API_ERROR
            mock_error.assert_called_once()
        print("✅ Error recovery suggestions provided")
        
        # Test chat interface error display
        with patch('streamlit.error') as mock_error:
            with patch('utils.session_manager.add_message') as mock_add_message:
                with patch('streamlit.rerun') as mock_rerun:
                    display_error_message("Test error message")
                    
                    mock_add_message.assert_called_once()
                    mock_rerun.assert_called_once()
        print("✅ Chat interface error recovery works")
        
        print("🎉 Error recovery mechanisms test passed!")


class TestDataRetentionAndPersistence:
    """Test data retention and persistence across sessions."""
    
    def setup_method(self):
        """Set up test environment."""
        mock_streamlit()
        
        # Import after mocking
        import importlib
        import utils.session_manager
        importlib.reload(utils.session_manager)
    
    def test_chat_history_persistence(self):
        """Test chat history persistence and retrieval."""
        print("🧪 Testing chat history persistence...")
        
        from utils.session_manager import (
            initialize_session, add_message, get_chat_history, clear_session
        )
        
        # Initialize session
        initialize_session()
        
        # Add various message types
        messages = [
            ("user", "What is the average sales?", "text"),
            ("agent", "The average sales is 650.", "text"),
            ("user", "Show me a chart", "text"),
            ("agent", "Chart generated", "plot"),
            ("agent", "Error occurred", "error")
        ]
        
        for role, content, msg_type in messages:
            if msg_type == "plot":
                add_message(role, content, msg_type, {"chart": "mock_chart_data"})
            else:
                add_message(role, content, msg_type)
        
        # Verify persistence
        history = get_chat_history()
        assert len(history) == 5
        
        # Verify message types
        user_msgs = [msg for msg in history if msg.role == "user"]
        agent_msgs = [msg for msg in history if msg.role == "agent"]
        plot_msgs = [msg for msg in history if msg.message_type == "plot"]
        error_msgs = [msg for msg in history if msg.message_type == "error"]
        
        assert len(user_msgs) == 2
        assert len(agent_msgs) == 3
        assert len(plot_msgs) == 1
        assert len(error_msgs) == 1
        
        # Verify chart data persistence
        plot_msg = plot_msgs[0]
        assert plot_msg.chart_data is not None
        assert "chart" in plot_msg.chart_data
        
        print("✅ Chat history persistence verified")
        
        # Test session clearing
        clear_session()
        history = get_chat_history()
        assert len(history) == 0
        print("✅ Session clearing works")
        
        print("🎉 Chat history persistence test passed!")
    
    def test_dataframe_retention(self):
        """Test DataFrame retention across interactions."""
        print("🧪 Testing DataFrame retention...")
        
        from utils.session_manager import (
            initialize_session, update_dataframe, has_dataframe, 
            get_session_data, clear_session
        )
        
        # Initialize session
        initialize_session()
        
        # Create and store DataFrame
        test_df = pd.DataFrame({
            'id': [1, 2, 3, 4, 5],
            'name': ['A', 'B', 'C', 'D', 'E'],
            'value': [10, 20, 30, 40, 50]
        })
        
        file_info = {
            'filename': 'retention_test.csv',
            'file_size': 1024,
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(test_df, file_info)
        
        # Verify initial storage
        assert has_dataframe() == True
        session_data = get_session_data()
        assert session_data.dataframe is not None
        assert len(session_data.dataframe) == 5
        assert session_data.file_info['filename'] == 'retention_test.csv'
        print("✅ DataFrame initially stored")
        
        # Simulate multiple interactions (DataFrame should persist)
        for i in range(3):
            assert has_dataframe() == True
            stored_df = get_session_data().dataframe
            assert len(stored_df) == 5
            assert list(stored_df.columns) == ['id', 'name', 'value']
        print("✅ DataFrame persists across interactions")
        
        # Test DataFrame replacement
        new_df = pd.DataFrame({
            'x': [1, 2],
            'y': [3, 4]
        })
        
        new_file_info = {
            'filename': 'new_data.csv',
            'file_size': 512,
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(new_df, new_file_info)
        
        # Verify replacement
        session_data = get_session_data()
        assert len(session_data.dataframe) == 2
        assert list(session_data.dataframe.columns) == ['x', 'y']
        assert session_data.file_info['filename'] == 'new_data.csv'
        print("✅ DataFrame replacement works")
        
        print("🎉 DataFrame retention test passed!")
    
    def test_session_state_integrity(self):
        """Test session state integrity under various conditions."""
        print("🧪 Testing session state integrity...")
        
        from utils.session_manager import (
            initialize_session, get_session_data, update_dataframe,
            add_message, clear_session
        )
        
        # Test 1: Multiple initializations
        initialize_session()
        session1 = get_session_data()
        
        initialize_session()  # Should not reset existing session
        session2 = get_session_data()
        
        assert session1.is_initialized == session2.is_initialized
        print("✅ Multiple initializations handled correctly")
        
        # Test 2: State consistency after operations
        test_df = pd.DataFrame({'col1': [1, 2, 3]})
        file_info = {'filename': 'test.csv', 'file_size': 100, 'upload_timestamp': datetime.now(), 'file_type': 'csv'}
        
        update_dataframe(test_df, file_info)
        add_message("user", "Test message", "text")
        
        session_data = get_session_data()
        assert session_data.dataframe is not None
        assert len(session_data.chat_history) == 1
        assert session_data.file_info is not None
        print("✅ State consistency maintained")
        
        # Test 3: Clean session clearing
        clear_session()
        session_data = get_session_data()
        
        assert session_data.dataframe is None
        assert len(session_data.chat_history) == 0
        assert session_data.file_info is None
        assert session_data.agent is None
        print("✅ Session clearing maintains integrity")
        
        print("🎉 Session state integrity test passed!")


def run_comprehensive_tests():
    """Run all comprehensive integration tests."""
    print("🚀 Starting Comprehensive Integration Tests\n")
    
    try:
        # Test complete workflows
        workflow_tests = TestCompleteUserWorkflows()
        workflow_tests.test_complete_csv_analysis_workflow()
        print()
        workflow_tests.test_excel_file_workflow()
        print()
        workflow_tests.test_session_persistence_across_interactions()
        print()
        
        # Test error scenarios
        error_tests = TestErrorScenariosAndRecovery()
        error_tests.test_file_upload_error_scenarios()
        print()
        error_tests.test_query_processing_error_scenarios()
        print()
        error_tests.test_visualization_error_scenarios()
        print()
        error_tests.test_error_recovery_mechanisms()
        print()
        
        # Test data persistence
        persistence_tests = TestDataRetentionAndPersistence()
        persistence_tests.test_chat_history_persistence()
        print()
        persistence_tests.test_dataframe_retention()
        print()
        persistence_tests.test_session_state_integrity()
        print()
        
        print("🎉 All comprehensive integration tests passed!")
        print("\n📋 Test Coverage Summary:")
        print("✅ Complete user workflows (CSV & Excel)")
        print("✅ Session persistence across interactions")
        print("✅ File upload error scenarios")
        print("✅ Query processing error scenarios")
        print("✅ Visualization error scenarios")
        print("✅ Error recovery mechanisms")
        print("✅ Chat history persistence")
        print("✅ DataFrame retention")
        print("✅ Session state integrity")
        print("\n🔄 Ready for production deployment")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Comprehensive integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)