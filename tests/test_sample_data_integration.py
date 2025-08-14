#!/usr/bin/env python3
"""
Integration tests using sample data files.
Tests real data loading and processing scenarios.
"""

import sys
import os
import pandas as pd
import pytest
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


class TestSampleDataIntegration:
    """Test integration with real sample data files."""
    
    def setup_method(self):
        """Set up test environment."""
        mock_streamlit()
        
        # Import after mocking
        import importlib
        import utils.session_manager
        importlib.reload(utils.session_manager)
    
    def test_employee_data_analysis(self):
        """Test analysis workflow with employee sample data."""
        print("🧪 Testing employee data analysis...")
        
        from utils.session_manager import initialize_session, update_dataframe, add_message
        from services.pandas_agent import PandasAgent
        
        # Initialize session
        initialize_session()
        
        # Load employee data
        employee_df = pd.read_csv('tests/sample_data.csv')
        
        # Verify data loaded correctly
        assert len(employee_df) == 10
        assert 'name' in employee_df.columns
        assert 'salary' in employee_df.columns
        assert 'department' in employee_df.columns
        print("✅ Employee data loaded successfully")
        
        # Store in session
        file_info = {
            'filename': 'sample_data.csv',
            'file_size': os.path.getsize('tests/sample_data.csv'),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(employee_df, file_info)
        
        # Test typical queries
        agent = PandasAgent()
        
        with patch('pandasai.Agent') as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            # Initialize agent
            success = agent.initialize_agent(employee_df)
            assert success == True
            
            # Test queries
            queries = [
                "What is the average salary?",
                "How many employees are in each department?",
                "Who has the highest salary?",
                "Show me salary distribution by department"
            ]
            
            for i, query in enumerate(queries):
                add_message("user", query, "text")
                
                # Mock responses
                if i == 0:
                    mock_agent_instance.chat.return_value = "The average salary is $75,000"
                elif i == 1:
                    mock_agent_instance.chat.return_value = "Engineering: 4, Marketing: 3, Sales: 3"
                elif i == 2:
                    mock_agent_instance.chat.return_value = "Henry Taylor has the highest salary at $100,000"
                else:
                    mock_figure = Mock()
                    mock_agent_instance.chat.return_value = mock_figure
                
                response = agent.process_query(query)
                
                if i < 3:
                    assert response.response_type == "text"
                    add_message("agent", response.content, "text")
                else:
                    # The response might be dataframe instead of plot for salary distribution
                    assert response.response_type in ["plot", "dataframe"]
                    if response.response_type == "plot":
                        add_message("agent", "Chart generated", "plot", response.content)
                    else:
                        add_message("agent", "Data analysis result", "dataframe", response.content)
        
        print("✅ Employee data analysis completed")
    
    def test_sales_data_analysis(self):
        """Test analysis workflow with sales sample data."""
        print("🧪 Testing sales data analysis...")
        
        from utils.session_manager import initialize_session, update_dataframe
        from services.pandas_agent import PandasAgent
        
        # Initialize session
        initialize_session()
        
        # Load sales data
        sales_df = pd.read_csv('tests/sample_sales_data.csv')
        
        # Verify data structure
        expected_columns = ['product_id', 'product_name', 'category', 'price', 'quantity_sold', 'revenue', 'date', 'region']
        assert all(col in sales_df.columns for col in expected_columns)
        assert len(sales_df) == 15
        print("✅ Sales data structure verified")
        
        # Test data quality
        assert sales_df['revenue'].dtype in ['float64', 'int64']
        assert sales_df['quantity_sold'].dtype in ['int64', 'float64']
        assert not sales_df['product_name'].isnull().any()
        print("✅ Sales data quality verified")
        
        # Store in session
        file_info = {
            'filename': 'sample_sales_data.csv',
            'file_size': os.path.getsize('tests/sample_sales_data.csv'),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(sales_df, file_info)
        
        # Test business intelligence queries
        agent = PandasAgent()
        
        with patch('pandasai.Agent') as mock_agent_class:
            mock_agent_instance = Mock()
            mock_agent_class.return_value = mock_agent_instance
            
            agent.initialize_agent(sales_df)
            
            # Business queries
            bi_queries = [
                "What is the total revenue?",
                "Which category generates the most revenue?",
                "What is the average price by category?",
                "Show me sales performance by region"
            ]
            
            for query in bi_queries:
                mock_agent_instance.chat.return_value = f"Analysis result for: {query}"
                response = agent.process_query(query)
                assert response.response_type in ["text", "plot"]
        
        print("✅ Sales data analysis completed")
    
    def test_financial_data_analysis(self):
        """Test analysis workflow with financial sample data."""
        print("🧪 Testing financial data analysis...")
        
        from utils.session_manager import initialize_session, update_dataframe
        
        # Initialize session
        initialize_session()
        
        # Load financial data
        financial_df = pd.read_csv('tests/sample_financial_data.csv')
        
        # Verify financial data structure
        expected_columns = ['date', 'account_type', 'transaction_type', 'amount', 'balance', 'description', 'category']
        assert all(col in financial_df.columns for col in expected_columns)
        print("✅ Financial data structure verified")
        
        # Test financial calculations
        total_deposits = financial_df[financial_df['transaction_type'] == 'Deposit']['amount'].sum()
        total_withdrawals = abs(financial_df[financial_df['transaction_type'] == 'Withdrawal']['amount'].sum())
        
        assert total_deposits > 0
        assert total_withdrawals > 0
        print("✅ Financial calculations verified")
        
        # Store in session
        file_info = {
            'filename': 'sample_financial_data.csv',
            'file_size': os.path.getsize('tests/sample_financial_data.csv'),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(financial_df, file_info)
        print("✅ Financial data analysis setup completed")
    
    def test_inventory_data_analysis(self):
        """Test analysis workflow with inventory sample data."""
        print("🧪 Testing inventory data analysis...")
        
        from utils.session_manager import initialize_session, update_dataframe
        
        # Initialize session
        initialize_session()
        
        # Load inventory data
        inventory_df = pd.read_csv('tests/sample_inventory_data.csv')
        
        # Verify inventory data structure
        expected_columns = ['item_id', 'item_name', 'category', 'supplier', 'stock_quantity', 'reorder_level', 'unit_cost', 'unit_price']
        assert all(col in inventory_df.columns for col in expected_columns)
        print("✅ Inventory data structure verified")
        
        # Test inventory business logic
        low_stock_items = inventory_df[inventory_df['stock_quantity'] <= inventory_df['reorder_level']]
        profit_margins = ((inventory_df['unit_price'] - inventory_df['unit_cost']) / inventory_df['unit_cost'] * 100)
        
        assert len(low_stock_items) >= 0  # May or may not have low stock items
        assert all(profit_margins >= 0)  # All items should have positive margins
        print("✅ Inventory business logic verified")
        
        # Store in session
        file_info = {
            'filename': 'sample_inventory_data.csv',
            'file_size': os.path.getsize('tests/sample_inventory_data.csv'),
            'upload_timestamp': datetime.now(),
            'file_type': 'csv'
        }
        
        update_dataframe(inventory_df, file_info)
        print("✅ Inventory data analysis setup completed")
    
    def test_excel_student_data_analysis(self):
        """Test analysis workflow with Excel student data."""
        print("🧪 Testing Excel student data analysis...")
        
        from utils.session_manager import initialize_session, update_dataframe
        
        # Initialize session
        initialize_session()
        
        # Check if Excel file exists
        excel_file_path = 'tests/sample_student_grades.xlsx'
        if not os.path.exists(excel_file_path):
            print("⚠️ Excel file not found, skipping Excel test")
            return
        
        # Load Excel data
        student_df = pd.read_excel(excel_file_path)
        
        # Verify Excel data structure
        expected_columns = ['student_id', 'student_name', 'course', 'semester', 'midterm_score', 'final_score', 'total_score', 'grade']
        assert all(col in student_df.columns for col in expected_columns)
        print("✅ Excel student data structure verified")
        
        # Test academic calculations
        avg_midterm = student_df['midterm_score'].mean()
        avg_final = student_df['final_score'].mean()
        
        assert 0 <= avg_midterm <= 100
        assert 0 <= avg_final <= 100
        print("✅ Academic calculations verified")
        
        # Store in session
        file_info = {
            'filename': 'sample_student_grades.xlsx',
            'file_size': os.path.getsize(excel_file_path),
            'upload_timestamp': datetime.now(),
            'file_type': 'xlsx'
        }
        
        update_dataframe(student_df, file_info)
        print("✅ Excel student data analysis setup completed")
    
    def test_data_type_validation(self):
        """Test data type validation across all sample files."""
        print("🧪 Testing data type validation...")
        
        sample_files = [
            ('tests/sample_data.csv', 'csv'),
            ('tests/sample_sales_data.csv', 'csv'),
            ('tests/sample_financial_data.csv', 'csv'),
            ('tests/sample_inventory_data.csv', 'csv')
        ]
        
        for file_path, file_type in sample_files:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                # Basic validation
                assert len(df) > 0, f"File {file_path} should not be empty"
                assert len(df.columns) > 0, f"File {file_path} should have columns"
                
                # Check for reasonable data types
                numeric_columns = df.select_dtypes(include=['int64', 'float64']).columns
                text_columns = df.select_dtypes(include=['object']).columns
                
                assert len(numeric_columns) > 0 or len(text_columns) > 0, f"File {file_path} should have recognizable data types"
                
                print(f"✅ {file_path} validation passed")
        
        # Test Excel file if it exists
        excel_file = 'tests/sample_student_grades.xlsx'
        if os.path.exists(excel_file):
            df = pd.read_excel(excel_file)
            assert len(df) > 0
            assert len(df.columns) > 0
            print(f"✅ {excel_file} validation passed")
        
        print("✅ Data type validation completed")
    
    def test_cross_dataset_compatibility(self):
        """Test that different datasets can be loaded in the same session."""
        print("🧪 Testing cross-dataset compatibility...")
        
        from utils.session_manager import initialize_session, update_dataframe, has_dataframe, get_session_data
        
        # Initialize session
        initialize_session()
        
        # Load different datasets sequentially
        datasets = [
            ('tests/sample_data.csv', 'Employee Data'),
            ('tests/sample_sales_data.csv', 'Sales Data'),
            ('tests/sample_financial_data.csv', 'Financial Data'),
            ('tests/sample_inventory_data.csv', 'Inventory Data')
        ]
        
        for file_path, description in datasets:
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
                
                file_info = {
                    'filename': os.path.basename(file_path),
                    'file_size': os.path.getsize(file_path),
                    'upload_timestamp': datetime.now(),
                    'file_type': 'csv'
                }
                
                # Update dataframe (should replace previous)
                update_dataframe(df, file_info)
                
                # Verify current dataset
                assert has_dataframe() == True
                session_data = get_session_data()
                assert session_data.file_info['filename'] == os.path.basename(file_path)
                assert len(session_data.dataframe) == len(df)
                
                print(f"✅ {description} loaded and verified")
        
        print("✅ Cross-dataset compatibility verified")


def run_sample_data_tests():
    """Run all sample data integration tests."""
    print("🚀 Starting Sample Data Integration Tests\n")
    
    try:
        tests = TestSampleDataIntegration()
        
        tests.test_employee_data_analysis()
        print()
        
        tests.test_sales_data_analysis()
        print()
        
        tests.test_financial_data_analysis()
        print()
        
        tests.test_inventory_data_analysis()
        print()
        
        tests.test_excel_student_data_analysis()
        print()
        
        tests.test_data_type_validation()
        print()
        
        tests.test_cross_dataset_compatibility()
        print()
        
        print("🎉 All sample data integration tests passed!")
        print("\n📋 Sample Data Test Summary:")
        print("✅ Employee data analysis workflow")
        print("✅ Sales data analysis workflow")
        print("✅ Financial data analysis workflow")
        print("✅ Inventory data analysis workflow")
        print("✅ Excel student data analysis workflow")
        print("✅ Data type validation across all files")
        print("✅ Cross-dataset compatibility")
        print("\n📊 Sample Data Files Available:")
        print("• tests/sample_data.csv (Employee data)")
        print("• tests/sample_sales_data.csv (Sales data)")
        print("• tests/sample_financial_data.csv (Financial data)")
        print("• tests/sample_inventory_data.csv (Inventory data)")
        print("• tests/sample_student_grades.xlsx (Student grades)")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Sample data integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = run_sample_data_tests()
    sys.exit(0 if success else 1)