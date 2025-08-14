#!/usr/bin/env python3
"""
Test case for maximum sales query to verify table name fixes.
"""

import pandas as pd
from services.pandas_agent import PandasAgent
import os

def test_maximum_sales_query():
    """Test the maximum sales query that was causing table name issues."""
    
    # Create sample data with sales column
    sample_data = {
        'product': ['Product A', 'Product B', 'Product C', 'Product D', 'Product E'],
        'sales': [1500, 2300, 1800, 2700, 2100],
        'region': ['North', 'South', 'East', 'West', 'Central'],
        'quarter': ['Q1', 'Q2', 'Q1', 'Q3', 'Q2']
    }

    df_copy = pd.DataFrame(sample_data)
    print('Sample Sales Data:')
    print(df_copy)
    print()

    # Test the enhanced PandasAgent
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('⚠️ GEMINI_API_KEY not found - cannot test with actual API')
        print('Please set GEMINI_API_KEY environment variable to test')
        return

    print('Initializing PandasAgent with sales data...')
    agent = PandasAgent(api_key)
    
    success = agent.initialize_agent(df_copy)
    if not success:
        print('❌ Failed to initialize agent')
        return

    print('✅ Agent initialized successfully')
    print()
    
    # Test the specific query that was causing issues
    test_query = 'What is the maximum sales from the table?'
    print(f'Testing query: "{test_query}"')
    print('Processing...')
    
    try:
        response = agent.process_query(test_query)
        
        print(f'Response Type: {response.response_type}')
        print(f'Response Content: {response.content}')
        print(f'Execution Time: {response.execution_time:.2f} seconds')
        
        if response.response_type == 'error':
            print(f'❌ Error occurred: {response.error_message}')
            
            # Check for specific table name issues
            error_msg = str(response.error_message).lower()
            if '<table_name>' in error_msg or 'expected table name' in error_msg:
                print('🔍 Table name placeholder issue detected')
                return False
            else:
                print('🔍 Different type of error (not table name related)')
                return False
        else:
            print('✅ Query executed successfully!')
            print(f'Maximum sales value: {response.content}')
            
            # Verify the result
            actual_max = df_copy['sales'].max()
            print(f'Expected maximum sales: {actual_max}')
            
            if str(actual_max) in str(response.content):
                print('✅ Result is correct!')
                return True
            else:
                print('⚠️ Result may not match expected value')
                return False
                
    except Exception as e:
        print(f'❌ Exception occurred: {str(e)}')
        
        # Check for table name issues in exception
        if '<table_name>' in str(e).lower():
            print('🔍 Table name placeholder issue in exception')
            return False
        else:
            print('🔍 Different type of exception')
            return False

if __name__ == "__main__":
    print("Testing Maximum Sales Query with Enhanced Table Name Handling")
    print("=" * 60)
    
    success = test_maximum_sales_query()
    
    print()
    print("=" * 60)
    if success:
        print("✅ TEST PASSED: Table name issues resolved!")
    else:
        print("❌ TEST FAILED: Table name issues still exist")
    print("Test completed.")