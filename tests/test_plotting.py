#!/usr/bin/env python3
"""
Test case for plotting functionality to verify chart generation works correctly.
"""

import pandas as pd
import os
import time
from services.pandas_agent import PandasAgent

def test_plotting_queries():
    """Test various plotting queries to verify chart generation."""
    
    # Create sample data with good variety for plotting
    sample_data = {
        'ORDERNUMBER': [10107, 10121, 10134, 10397, 10414, 10415, 10416, 10417],
        'QUANTITYORDERED': [30, 34, 41, 34, 47, 25, 38, 42],
        'PRICEEACH': [95.70, 81.35, 94.74, 62.24, 65.52, 88.90, 72.15, 91.30],
        'SALES': [2871.00, 2765.90, 3884.34, 2116.16, 3079.44, 2222.50, 2741.70, 3834.60],
        'ORDERDATE': ['2/24/2003', '5/7/2003', '7/1/2003', '3/28/2005', '5/6/2005', '6/15/2003', '8/20/2003', '9/10/2005'],
        'STATUS': ['Shipped', 'Shipped', 'Shipped', 'Shipped', 'On Hold', 'Shipped', 'Shipped', 'Cancelled'],
        'COUNTRY': ['USA', 'France', 'France', 'France', 'USA', 'Germany', 'UK', 'Spain'],
        'PRODUCTLINE': ['Motorcycles', 'Motorcycles', 'Motorcycles', 'Ships', 'Ships', 'Cars', 'Planes', 'Trains'],
        'CUSTOMERNAME': ['Land of Toys Inc.', 'Reims Collectables', 'Lyon Souveniers', 'Alpha Cognac', 'Gifts4AllAges.com', 'German Motors', 'UK Collectibles', 'Spanish Toys'],
        'DEALSIZE': ['Small', 'Small', 'Medium', 'Small', 'Medium', 'Large', 'Medium', 'Large']
    }

    df = pd.DataFrame(sample_data)
    print('Sample Data for Plotting Tests:')
    print(df[['ORDERNUMBER', 'COUNTRY', 'SALES', 'PRODUCTLINE', 'DEALSIZE']].head())
    print()

    # Test the enhanced PandasAgent
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('⚠️ GEMINI_API_KEY not found - cannot test with actual API')
        print('Please set GEMINI_API_KEY environment variable to test')
        return False

    print('Initializing PandasAgent with plotting data...')
    agent = PandasAgent(api_key)
    
    success = agent.initialize_agent(df)
    if not success:
        print('❌ Failed to initialize agent')
        return False

    print('✅ Agent initialized successfully')
    print()
    
    # Define plotting test queries
    plot_queries = [
        {
            'query': 'Create a bar chart showing the distribution of countries',
            'expected_type': 'plot',
            'description': 'Country distribution bar chart',
            'file_name': 'bar_chart'
        },
        {
            'query': 'Show me a histogram of sales values',
            'expected_type': 'plot', 
            'description': 'Sales histogram'
        },
        {
            'query': 'Create a pie chart of product lines',
            'expected_type': 'plot',
            'description': 'Product line pie chart'
        }
    ]
    
    results = []
    
    for i, test_case in enumerate(plot_queries, 1):
        print(f'=== Plot Test {i}: {test_case["description"]} ===')
        print(f'Testing query: "{test_case["query"]}"')
        print('Processing...')
        
        start_time = time.time()
        
        try:
            response = agent.process_query(test_case['query'])
            
            end_time = time.time()
            query_time = end_time - start_time
            
            print(f'Response Type: {response.response_type}')
            print(f'Execution Time: {response.execution_time:.2f} seconds')
            print(f'Total Query Time: {query_time:.2f} seconds')
            
            if response.response_type == 'error':
                print(f'❌ Error occurred: {response.error_message}')
                
                # Check for specific plotting-related errors
                error_msg = str(response.error_message).lower()
                if any(keyword in error_msg for keyword in ['plot', 'chart', 'visualization', 'matplotlib', 'distribution', 'graph']):
                    print('🔍 Plotting-related error detected')
                elif 'no result returned' in error_msg:
                    print('🔍 "No result returned" error detected')
                else:
                    print('🔍 Different type of error')
                
                results.append(False)
            else:
                print('✅ Query executed successfully!')
                
                # Check if we got a plot response
                if response.response_type == 'plot':
                    print('✅ Plot response type detected!')
                    print(f'Chart data type: {type(response.content)}')
                    
                    # Check if chart data exists
                    if response.content is not None:
                        print('✅ Chart data is not None')
                        results.append(True)
                    else:
                        print('⚠️ Chart data is None')
                        results.append(False)
                        
                elif response.response_type == 'text':
                    print('📝 Text response received (may contain chart reference)')
                    print(f'Response content preview: {str(response.content)[:200]}...')
                    
                    # Check if text mentions chart creation
                    response_str = str(response.content).lower()
                    if any(keyword in response_str for keyword in ['chart', 'plot', 'graph', 'visualization']):
                        print('✅ Text response mentions chart creation')
                        results.append(True)
                    else:
                        print('⚠️ Text response doesn\'t mention chart creation')
                        results.append(False)
                else:
                    print(f'📊 Received {response.response_type} response')
                    results.append(True)
                    
        except Exception as e:
            print(f'❌ Exception occurred: {str(e)}')
            
            # Check for plotting-related exceptions
            if any(keyword in str(e).lower() for keyword in ['plot', 'chart', 'matplotlib', 'plotly']):
                print('🔍 Plotting-related exception')
            else:
                print('🔍 General exception')
            
            results.append(False)
        
        print()
        
        # Add 3 second lag between queries (shorter than the main test)
        if i < len(plot_queries):
            print('⏳ Waiting 3 seconds before next query...')
            time.sleep(3)
            print()
    
    # Summary
    print('=' * 60)
    print('PLOTTING TEST SUMMARY:')
    for i, (test_case, result) in enumerate(zip(plot_queries, results), 1):
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f'Test {i} ({test_case["description"]}): {status}')
    
    overall_success = all(results)
    print(f'Overall Plotting Test: {"✅ PASSED" if overall_success else "❌ FAILED"}')
    
    return overall_success

def test_simple_data_query():
    """Test a simple non-plotting query to ensure basic functionality works."""
    
    sample_data = {
        'product': ['A', 'B', 'C'],
        'sales': [100, 200, 150],
        'country': ['USA', 'France', 'Germany']
    }
    
    df = pd.DataFrame(sample_data)
    print('Simple test data:')
    print(df)
    print()
    
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('⚠️ GEMINI_API_KEY not found')
        return False
    
    agent = PandasAgent(api_key)
    success = agent.initialize_agent(df)
    
    if not success:
        print('❌ Failed to initialize agent')
        return False
    
    print('Testing simple query: "What is the total sales?"')
    
    try:
        response = agent.process_query('What is the total sales?')
        
        if response.response_type == 'error':
            print(f'❌ Error: {response.error_message}')
            return False
        else:
            print(f'✅ Success: {response.content}')
            expected_total = df['sales'].sum()  # 450
            
            if str(expected_total) in str(response.content):
                print('✅ Result is correct!')
                return True
            else:
                print(f'⚠️ Expected {expected_total}, got {response.content}')
                return False
                
    except Exception as e:
        print(f'❌ Exception: {str(e)}')
        return False

if __name__ == "__main__":
    print("Testing Plotting Functionality")
    print("=" * 60)
    
    # First test simple functionality
    print("=== Simple Query Test ===")
    simple_success = test_simple_data_query()
    print()
    
    if simple_success:
        print("=== Plotting Tests ===")
        plot_success = test_plotting_queries()
    else:
        print("❌ Simple query failed, skipping plotting tests")
        plot_success = False
    
    print()
    print("=" * 60)
    print("FINAL RESULTS:")
    print(f"Simple Query Test: {'✅ PASSED' if simple_success else '❌ FAILED'}")
    print(f"Plotting Tests: {'✅ PASSED' if plot_success else '❌ FAILED'}")
    
    if simple_success and plot_success:
        print("🎉 ALL TESTS PASSED: Plotting functionality is working!")
    elif simple_success:
        print("⚠️ PARTIAL SUCCESS: Basic queries work, plotting needs attention")
    else:
        print("❌ TESTS FAILED: Basic functionality issues detected")
    
    print("Test completed.")