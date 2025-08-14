#!/usr/bin/env python3
"""
Test case for average price query to verify "No code found" error handling.
"""

import pandas as pd
from services.pandas_agent import PandasAgent
import os

def test_average_price_query():
    """Test the average price query that was causing 'No code found' issues."""
    
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
        return

    print('Initializing PandasAgent with sales data...')
    agent = PandasAgent(api_key)
    
    success = agent.initialize_agent(df)
    if not success:
        print('❌ Failed to initialize agent')
        return

    print('✅ Agent initialized successfully')
    print()
    
    # Test the specific query that was causing issues
    test_query = 'what is the maximum priceeach?'
    print(f'Testing query: "{test_query}"')
    print('Processing...')
    
    try:
        response = agent.process_query(test_query)
        
        print(f'Response Type: {response.response_type}')
        print(f'Response Content: {response.content}')
        print(f'Execution Time: {response.execution_time:.2f} seconds')
        
        if response.response_type == 'error':
            print(f'❌ Error occurred: {response.error_message}')
            
            # Check for specific "No code found" issues
            error_msg = str(response.error_message).lower()
            if 'no code found' in error_msg or 'code' in error_msg:
                print('🔍 "No code found" issue detected')
                return False
            else:
                print('🔍 Different type of error (not code generation related)')
                return False
        else:
            print('✅ Query executed successfully!')
            print(f'Average price value: {response.content}')
            
            # Verify the result
            actual_avg = df['PRICEEACH'].mean()
            print(f'Expected average price: {actual_avg:.2f}')
            
            # Check if the result contains the expected value (allowing for some formatting differences)
            response_str = str(response.content)
            if f"{actual_avg:.2f}" in response_str or f"{actual_avg:.1f}" in response_str:
                print('✅ Result is correct!')
                return True
            else:
                print('⚠️ Result may not match expected value')
                print(f'Response contains: {response_str}')
                return False
                
    except Exception as e:
        print(f'❌ Exception occurred: {str(e)}')
        
        # Check for "No code found" issues in exception
        if 'no code found' in str(e).lower():
            print('🔍 "No code found" issue in exception')
            return False
        else:
            print('🔍 Different type of exception')
            return False

if __name__ == "__main__":
    print("Testing Average Price Query with Enhanced Error Handling")
    print("=" * 60)
    
    success = test_average_price_query()
    
    print()
    print("=" * 60)
    if success:
        print("✅ TEST PASSED: 'No code found' issues resolved!")
    else:
        print("❌ TEST FAILED: 'No code found' issues still exist")
    print("Test completed.")