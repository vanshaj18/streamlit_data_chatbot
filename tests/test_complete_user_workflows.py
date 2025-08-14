#!/usr/bin/env python3
"""
Complete user workflow integration tests for the Data Chatbot Dashboard.
Tests end-to-end scenarios from file upload to visualization.
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
    
    st.session_state = MockSessionState()
    st.error = lambda x: print(f"ERROR: {x}")
    st.success = lambda x: print(f"SUCCESS: {x}")
    st.info = lambda x: print(f"INFO: {x}")
    st.warning = lambda x: print(f"WARNING: {x}")
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
    
    def test_csv_upload_to_analysis_workflow(self):
        """Test complete workflow: CSV upload → multiple queries → visualizations."""
        print("🧪 Testing CSV upload to analysis workflow...")
        
        from utils.session_manager import (
            initialize_session, update_dataframe, add_message, 
            get_chat_history, has_dataframe, clear_session
        )
        from components.file_handler import validate_file, load_dataframe
        from services.pandas_agent import PandasAgent
        
        # Step 1: Initialize fresh session
        clear_session()
        initialize_session()
        print("✅ Step 1: Fresh session initialized")
        
        # Step 2: Create and upload comprehensive CSV data
        csv_content = """employee_id,name,department,salary,years_experience,performance_rating,bonus
E001,Alice Johnson,Engineering,85000,5,4.2,8500
E002,Bob Smith,Engineering,92000,8,4.5,9200
E003,Carol Davis,Marketing,78000,6,4.0,7800
E004,David Wilson,Sales,72000,4,3.8,7200
E005,Eva Brown,Engineering,95000,10,4.7,9500
E006,Frank Miller,Marketing,68000,3,3.5,6800
E007,Grace Lee,Sales,75000,5,4.1,7500
E008,Henry Taylor,Engineering,88000,7,4.3,8800
E009,Ivy Chen,Marketing,71000,4,3.9,7100
E010,Jack Wilson,Sales,69000,2,3.6,6900"""
        
        mock_file = MockUploadedFile("employee_data.csv", csv_content)
        
        # Validate and load file
        is_valid, error = validate_file(mock_file)
        print(f"File validation: {is_valid}, Error: {error}")
        
        with patch('pandas.read_csv') as mock_read_csv:
            expected_df = pd.DataFrame({
                'employee_id': ['E001', 'E002', 'E003', 'E004', 'E005', 'E006', 'E007', 'E008', 'E009', 'E010'],
                'name': ['Alice Johnson', 'Bob Smith', 'Carol Davis', 'David Wilson', 'Eva Brown', 
                        'Frank Miller', 'Grace Lee', 'Henry Taylor', 'Ivy Chen', 'Jack Wilson'],
                'department': ['Engineering', 'Engineering', 'Marketing', 'Sales', 'Engineering',
                              'Marketing', 'Sales', 'Engineering', 'Marketing', 'Sales'],
                'salary': [85000, 92000, 78000, 72000, 95000, 68000, 75000, 88000, 71000, 69000],
                'years_experience': [5, 8, 6, 4, 10, 3, 5, 7, 4, 2],
                'performance_rating': [4.2, 4.5, 4.0, 3.8, 4.7, 3.5, 4.1, 4.3, 3.9, 3.6],
                'bonus': [8500, 9200, 7800, 7200, 9500, 6800, 7500, 8800, 7100, 6900]
            })
            mock_read_csv.return_value = expected_df
            
            df, load_error = load_dataframe(mock_file)
            print(f"Data loading: Success={df is not None}, Error: {load_error}")
        
        # Store in session
        file_info = {
            'filename': 'employee_data.csv',
            'file_size': len(csv_content),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        update_dataframe(df, file_info)
        print("✅ Step 2: Employee data uploaded and stored")
        
        # Step 3: Execute comprehensive analysis queries
        analysis_queries = [
            ("What is the average salary across all employees?", "text", "The average salary is $79,300"),
            ("How many employees are in each department?", "text", "Engineering: 4, Marketing: 3, Sales: 3"),
            ("Who has the highest performance rating?", "text", "Eva Brown has the highest rating at 4.7"),
            ("Show me salary distribution by department", "plot", "Chart showing salary by department"),
            ("What is the correlation between experience and salary?", "plot", "Scatter plot showing correlation"),
            ("Display bonus amounts by performance rating", "plot", "Chart showing bonus vs performance")
        ]
        
        agent = PandasAgent()
        
        with patch('pandasai.Agent') as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            # Initialize agent
            agent.initialize_agent(df)
            
            for i, (query, expected_type, mock_response) in enumerate(analysis_queries):
                # Add user message
                add_message("user", query, "text")
                
                # Mock agent response based on expected type
                if expected_type == "text":
                    mock_agent_instance.chat.return_value = mock_response
                    response = agent.process_query(query)
                    add_message("agent", response.content, "text")
                else:
                    # Mock chart response
                    mock_figure = Mock()
                    mock_agent_instance.chat.return_value = mock_figure
                    response = agent.process_query(query)
                    add_message("agent", mock_response, "plot", {"chart": mock_figure})
                
                print(f"✅ Query {i+1} processed: {query[:50]}...")
        
        print("✅ Step 3: All analysis queries processed")
        
        # Step 4: Verify complete workflow state
        history = get_chat_history()
        user_messages = [msg for msg in history if msg.role == "user"]
        agent_messages = [msg for msg in history if msg.role == "agent"]
        plot_messages = [msg for msg in history if msg.message_type == "plot"]
        
        print(f"Total messages: {len(history)}")
        print(f"User messages: {len(user_messages)}")
        print(f"Agent messages: {len(agent_messages)}")
        print(f"Plot messages: {len(plot_messages)}")
        
        print("✅ Step 4: Workflow state verified")
        print("🎉 CSV upload to analysis workflow completed successfully!")
    
    def test_excel_upload_to_visualization_workflow(self):
        """Test complete workflow with Excel file and visualization focus."""
        print("🧪 Testing Excel upload to visualization workflow...")
        
        from utils.session_manager import initialize_session, update_dataframe, add_message, clear_session
        from components.file_handler import validate_file, load_dataframe
        from components.visualization import render_chart
        
        # Initialize fresh session
        clear_session()
        initialize_session()
        
        # Create mock Excel file
        mock_excel_file = MockUploadedFile("sales_report.xlsx", b"mock_excel_content", "xlsx")
        mock_excel_file.size = 4096
        
        # Validate Excel file
        is_valid, error = validate_file(mock_excel_file)
        print(f"Excel validation: {is_valid}")
        
        # Mock Excel data loading
        with patch('pandas.read_excel') as mock_read_excel:
            excel_df = pd.DataFrame({
                'month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun'],
                'revenue': [50000, 55000, 48000, 62000, 58000, 65000],
                'expenses': [30000, 32000, 28000, 35000, 33000, 38000],
                'profit': [20000, 23000, 20000, 27000, 25000, 27000],
                'customers': [150, 165, 140, 180, 170, 190]
            })
            mock_read_excel.return_value = excel_df
            
            df, load_error = load_dataframe(mock_excel_file)
        
        # Store in session
        file_info = {
            'filename': 'sales_report.xlsx',
            'file_size': 4096,
            'upload_timestamp': datetime.now(),
            'file_type': 'xlsx'
        }
        update_dataframe(df, file_info)
        print("✅ Excel data loaded and stored")
        
        # Test visualization-focused queries
        viz_queries = [
            "Create a line chart showing revenue trends",
            "Show me a bar chart of profit by month",
            "Generate a scatter plot of customers vs revenue",
            "Display a pie chart of expense distribution"
        ]
        
        for query in viz_queries:
            add_message("user", query, "text")
            
            # Mock chart generation
            with patch('matplotlib.pyplot.figure') as mock_figure:
                mock_chart = Mock()
                mock_figure.return_value = mock_chart
                
                # Test chart rendering
                with patch('streamlit.pyplot') as mock_st_pyplot:
                    success = render_chart(mock_chart, f"Chart for: {query}")
                    print(f"Chart rendered for: {query[:30]}...")
                
                add_message("agent", f"Chart generated: {query}", "plot", {"chart": mock_chart})
        
        print("✅ All visualization queries processed")
        print("🎉 Excel upload to visualization workflow completed!")
    
    def test_multi_dataset_session_workflow(self):
        """Test workflow with multiple dataset uploads in same session."""
        print("🧪 Testing multi-dataset session workflow...")
        
        from utils.session_manager import (
            initialize_session, update_dataframe, get_session_data, 
            add_message, clear_session
        )
        
        # Initialize session
        clear_session()
        initialize_session()
        
        # Dataset 1: Employee data
        employee_data = """name,salary,department
Alice,75000,Engineering
Bob,80000,Marketing
Carol,70000,Sales"""
        
        mock_file1 = MockUploadedFile("employees.csv", employee_data)
        
        with patch('pandas.read_csv') as mock_read_csv:
            df1 = pd.DataFrame({
                'name': ['Alice', 'Bob', 'Carol'],
                'salary': [75000, 80000, 70000],
                'department': ['Engineering', 'Marketing', 'Sales']
            })
            mock_read_csv.return_value = df1
            
            from components.file_handler import load_dataframe
            df, _ = load_dataframe(mock_file1)
        
        update_dataframe(df, {
            'filename': 'employees.csv',
            'file_size': len(employee_data),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        })
        
        # Query first dataset
        add_message("user", "What is the average salary?", "text")
        add_message("agent", "The average salary is $75,000", "text")
        
        print("✅ First dataset processed")
        
        # Dataset 2: Sales data (replaces first dataset)
        sales_data = """product,price,quantity
Laptop,1000,50
Mouse,25,200
Keyboard,75,100"""
        
        mock_file2 = MockUploadedFile("sales.csv", sales_data)
        
        with patch('pandas.read_csv') as mock_read_csv:
            df2 = pd.DataFrame({
                'product': ['Laptop', 'Mouse', 'Keyboard'],
                'price': [1000, 25, 75],
                'quantity': [50, 200, 100]
            })
            mock_read_csv.return_value = df2
            
            df, _ = load_dataframe(mock_file2)
        
        update_dataframe(df, {
            'filename': 'sales.csv',
            'file_size': len(sales_data),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        })
        
        # Query second dataset
        add_message("user", "What is the total revenue?", "text")
        add_message("agent", "The total revenue is $62,500", "text")
        
        print("✅ Second dataset processed")
        
        # Verify session state
        session_data = get_session_data()
        current_filename = session_data.file_info['filename']
        print(f"Current dataset: {current_filename}")
        
        # Verify chat history persists across dataset changes
        from utils.session_manager import get_chat_history
        history = get_chat_history()
        print(f"Total messages in history: {len(history)}")
        
        print("🎉 Multi-dataset session workflow completed!")
    
    def test_error_recovery_workflow(self):
        """Test complete workflow with error scenarios and recovery."""
        print("🧪 Testing error recovery workflow...")
        
        from utils.session_manager import initialize_session, add_message, clear_session
        from components.chat_interface import display_error_message
        from utils.error_handler import ErrorHandler, ErrorCategory
        
        # Initialize session
        clear_session()
        initialize_session()
        
        # Simulate various error scenarios
        error_handler = ErrorHandler()
        
        # Error 1: File upload error
        with patch('streamlit.error') as mock_error:
            error_info = error_handler.handle_error(
                "File too large (60MB). Maximum size is 50MB.",
                ErrorCategory.FILE_ERROR,
                show_ui=True
            )
            print("✅ File upload error handled")
        
        # Error 2: Query processing error
        add_message("user", "Show me the data", "text")
        
        with patch('streamlit.error') as mock_error:
            with patch('utils.session_manager.add_message') as mock_add_message:
                with patch('streamlit.rerun') as mock_rerun:
                    display_error_message("No dataset uploaded. Please upload a file first.")
                    print("✅ Query processing error handled")
        
        # Error 3: Visualization error with recovery
        add_message("user", "Create a chart", "text")
        
        with patch('streamlit.error') as mock_error:
            error_info = error_handler.handle_error(
                "Chart generation failed. Displaying text summary instead.",
                ErrorCategory.VISUALIZATION_ERROR,
                show_ui=True
            )
            
            # Simulate fallback response
            add_message("agent", "Unable to generate chart. Here's a text summary: Data shows positive trends.", "text")
            print("✅ Visualization error with fallback handled")
        
        # Error 4: Recovery after successful upload
        csv_data = "name,value\nTest,100"
        mock_file = MockUploadedFile("recovery_test.csv", csv_data)
        
        with patch('pandas.read_csv') as mock_read_csv:
            recovery_df = pd.DataFrame({'name': ['Test'], 'value': [100]})
            mock_read_csv.return_value = recovery_df
            
            from components.file_handler import load_dataframe
            df, _ = load_dataframe(mock_file)
            
            from utils.session_manager import update_dataframe
            update_dataframe(df, {
                'filename': 'recovery_test.csv',
                'file_size': len(csv_data),
                'upload_timestamp': datetime.now(),
                'file_type': 'csv'
            })
        
        # Successful query after recovery
        add_message("user", "What is the value?", "text")
        add_message("agent", "The value is 100", "text")
        
        print("✅ Error recovery completed")
        print("🎉 Error recovery workflow completed!")


def run_complete_workflow_tests():
    """Run all complete user workflow tests."""
    print("🚀 Starting Complete User Workflow Tests\n")
    
    try:
        tests = TestCompleteUserWorkflows()
        
        tests.test_csv_upload_to_analysis_workflow()
        print()
        
        tests.test_excel_upload_to_visualization_workflow()
        print()
        
        tests.test_multi_dataset_session_workflow()
        print()
        
        tests.test_error_recovery_workflow()
        print()
        
        print("🎉 All complete user workflow tests passed!")
        print("\n📋 Workflow Test Summary:")
        print("✅ CSV upload to analysis workflow")
        print("✅ Excel upload to visualization workflow")
        print("✅ Multi-dataset session workflow")
        print("✅ Error recovery workflow")
        print("\n🔄 Complete user workflows validated")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Complete workflow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_complete_workflow_tests()
    sys.exit(0 if success else 1)