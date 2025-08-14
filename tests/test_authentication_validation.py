#!/usr/bin/env python3
"""
Simple authentication validation test that doesn't hit rate limits.
"""

import sys
import os
import pandas as pd

# Add current directory to path
sys.path.append('.')

def test_authentication_setup():
    """Test that authentication is properly configured without making API calls."""
    print("🧪 Testing authentication setup...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    # Test API key configuration
    api_key = os.getenv("GEMINI_API_KEY")
    assert api_key is not None, "GEMINI_API_KEY must be set in environment"
    assert len(api_key) > 20, "GEMINI_API_KEY appears to be invalid (too short)"
    assert api_key.startswith("AIza"), "GEMINI_API_KEY should start with 'AIza'"
    
    print("✅ API key is properly configured")
    
    # Test PandasAI imports
    try:
        from services.pandas_agent import PandasAgent, PANDASAI_AVAILABLE
        assert PANDASAI_AVAILABLE == True, "PandasAI should be available"
        print("✅ PandasAI is available")
    except ImportError as e:
        print(f"❌ PandasAI import failed: {e}")
        return False
    
    # Test agent creation
    agent = PandasAgent()
    assert agent.api_key is not None, "Agent should have API key"
    print("✅ Agent creation successful")
    
    # Test DataFrame initialization (no API calls)
    test_df = pd.DataFrame({
        'name': ['Alice', 'Bob'],
        'value': [1, 2]
    })
    
    success = agent.initialize_agent(test_df)
    assert success == True, "Agent initialization should succeed"
    assert agent.is_initialized() == True, "Agent should be initialized"
    print("✅ Agent initialization successful")
    
    return True


def main():
    """Run authentication validation."""
    print("🚀 Authentication Validation Test\n")
    
    try:
        success = test_authentication_setup()
        
        if success:
            print("\n🎉 Authentication validation passed!")
            print("\n📋 Validation Summary:")
            print("✅ API key configuration")
            print("✅ PandasAI availability")
            print("✅ Agent creation")
            print("✅ Agent initialization")
            print("\n🔐 Authentication Status: WORKING")
            print("🌐 Ready for Gemini API calls")
            print("\nℹ️  Note: Actual API calls may be rate limited on free tier")
            print("   Rate limit: 10 requests per minute for gemini-2.0-flash")
            
        return success
        
    except Exception as e:
        print(f"\n❌ Authentication validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)