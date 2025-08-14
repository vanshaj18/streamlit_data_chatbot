#!/usr/bin/env python3
"""
Test case for running two subsequent queries with 5 second lag to test session persistence and robustness.
"""

import pandas as pd
from services.pandas_agent import PandasAgent
import os
import time

def test_double_query_with_lag():
    """Test two subsequent queries with 5 second lag."""
    
    # Create sample data similar to the one mentioned in the error
    sample_data = {
        'ORDERNUMBER': [10107, 10121, 10134, 10397, 10414],
        'QUANTITYORDERED': [30, 34, 41, 34, 47],
        'PRICEEACH': [95.70, 81.35, 94.74, 62.24, 65.52],
        'ORDERLINENUMBER': [2, 5, 2, 1, 9],
        'SALES': [2871.00, 2765.90, 3884.34, 2116.16, 3079.44],
        'ORDERDATE': ['2/24/2003', '5/7/2003', '7/1/2003', '3/28/2005', '5/6/2005'],
        'STATUS': ['Shipped', 'Shipped', 'Shipped', 'Shipped', 'On Hold'],
        'QTR_ID': [1, 2, 3, 1, 2],
        'MONTH_ID': [2, 5, 7, 3, 5],
        'YEAR_ID': [2003, 2003, 2003, 2005, 2005],
        'PRODUCTLINE': ['Motorcycles', 'Motorcycles', 'Motorcycles', 'Ships', 'Ships'],
        'MSRP': [95, 95, 95, 54, 54],
        'PRODUCTCODE': ['S10_1678', 'S10_1678', 'S10_1678', 'S72_3212', 'S72_3212'],
        'CUSTOMERNAME': ['Land of Toys Inc.', 'Reims Collectables', 'Lyon Souveniers', 'Alpha Cognac', 'Gifts4AllAges.com'],
        'PHONE': ['2125557818', '26.47.1555', '+33 1 46 62 7555', '61.77.6555', '6175559555'],
        'ADDRESSLINE1': ['897 Long Airport Avenue', '59 rue de l\'Abbaye', '27 rue du Colonel Pierre Avia', '1 rue Alsace-Lorraine', '8616 Spinnaker Dr.'],
        'CITY': ['NYC', 'Reims', 'Paris', 'Toulouse', 'Boston'],
        'STATE': ['NY', None, None, None, 'MA'],
        'POSTALCODE': ['10022', '51100', '75508', '31000', '51003'],
        'COUNTRY': ['USA', 'France', 'France', 'France', 'USA'],
        'TERRITORY': [None, 'EMEA', 'EMEA', 'EMEA', None],
        'CONTACTLASTNAME': ['Yu', 'Henriot', 'Da Cunha', 'Roulet', 'Yoshido'],
        'CONTACTFIRSTNAME': ['Kwai', 'Paul', 'Daniel', 'Annette', 'Juri'],
        'DEALSIZE': ['Small', 'Small', 'Medium', 'Small', 'Medium']
    }

    df = pd.DataFrame(sample_data)
    print('Sample Sales Data:')
    print(df[['ORDERNUMBER', 'PRICEEACH', 'SALES', 'CUSTOMERNAME']].head())
    print()

    # Test the enhanced PandasAgent
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('⚠️ GEMINI_API_KEY not found - cannot test with actual API')
        print('Please set GEMINI_API_KEY environment variable to test')
        return False

    print('Initializing PandasAgent with sales data...')
    agent = PandasAgent(api_key)
    
    success = agent.initialize_agent(df)
    if not success:
        print('❌ Failed to initialize agent')
        return False

    print('✅ Agent initialized successfully')
    print()
    
    # Define two test queries
    queries = [
        'what is the average priceeach?',
        'what is the maximum sales value?'
    ]
    
    expected_results = [
        df['PRICEEACH'].mean(),  # Expected average price
        df['SALES'].max()        # Expected maximum sales
    ]
    
    results = []
    
    for i, (query, expected) in enumerate(zip(queries, expected_results), 1):
        print(f'=== Query {i} ===')
        print(f'Testing query: "{query}"')
        print('Processing...')
        
        start_time = time.time()
        
        try:
            response = agent.process_query(query)
            
            end_time = time.time()
            query_time = end_time - start_time
            
            print(f'Response Type: {response.response_type}')
            print(f'Response Content: {response.content}')
            print(f'Execution Time: {response.execution_time:.2f} seconds')
            print(f'Total Query Time: {query_time:.2f} seconds')
            
            if response.response_type == 'error':
                print(f'❌ Error occurred: {response.error_message}')
                results.append(False)
            else:
                print('✅ Query executed successfully!')
                
                # Verify the result
                print(f'Expected result: {expected:.2f}')
                
                # Check if the result contains the expected value (allowing for some formatting differences)
                response_str = str(response.content)
                if f"{expected:.2f}" in response_str or f"{expected:.1f}" in response_str or f"{int(expected)}" in response_str:
                    print('✅ Result is correct!')
                    results.append(True)
                else:
                    print('⚠️ Result may not match expected value')
                    print(f'Response contains: {response_str}')
                    results.append(False)
                    
        except Exception as e:
            print(f'❌ Exception occurred: {str(e)}')
            results.append(False)
        
        print()
        
        # Add 5 second lag between queries (except after the last query)
        if i < len(queries):
            print('⏳ Waiting 5 seconds before next query...')
            time.sleep(5)
            print()
    
    # Summary
    print('=' * 60)
    print('SUMMARY:')
    print(f'Query 1 (Average Price): {"✅ PASSED" if results[0] else "❌ FAILED"}')
    print(f'Query 2 (Maximum Sales): {"✅ PASSED" if results[1] else "❌ FAILED"}')
    
    overall_success = all(results)
    print(f'Overall Test: {"✅ PASSED" if overall_success else "❌ FAILED"}')
    
    return overall_success

if __name__ == "__main__":
    print("Testing Double Query with 5 Second Lag")
    print("=" * 60)
    
    success = test_double_query_with_lag()
    
    print()
    print("=" * 60)
    if success:
        print("✅ ALL TESTS PASSED: Session persistence and robustness confirmed!")
    else:
        print("❌ SOME TESTS FAILED: Issues detected with session persistence or query handling")
    print("Test completed.")