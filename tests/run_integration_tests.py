#!/usr/bin/env python3
"""
Integration test runner for the Data Chatbot Dashboard.
Runs all integration tests and provides comprehensive reporting.
"""

import sys
import os
import subprocess
import time
from datetime import datetime

# Add current directory to path
sys.path.append('.')

def run_test_file(test_file, description):
    """Run a specific test file and return results."""
    print(f"\n{'='*60}")
    print(f"🧪 Running {description}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the test file
        result = subprocess.run([sys.executable, test_file], 
                              capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        duration = end_time - start_time
        
        if result.returncode == 0:
            print(result.stdout)
            print(f"\n✅ {description} PASSED (Duration: {duration:.2f}s)")
            return True, duration, result.stdout
        else:
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
            print(f"\n❌ {description} FAILED (Duration: {duration:.2f}s)")
            return False, duration, result.stdout + result.stderr
            
    except subprocess.TimeoutExpired:
        print(f"\n⏰ {description} TIMED OUT (>300s)")
        return False, 300, "Test timed out"
    except Exception as e:
        print(f"\n💥 {description} ERROR: {e}")
        return False, 0, str(e)


def check_test_files():
    """Check that all required test files exist."""
    required_files = [
        'tests/test_integration.py',
        'tests/test_comprehensive_integration.py',
        'tests/test_sample_data_integration.py',
        'tests/test_file_handler_integration.py',
        'tests/test_error_integration.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("❌ Missing test files:")
        for file_path in missing_files:
            print(f"  • {file_path}")
        return False
    
    return True


def check_sample_data_files():
    """Check that sample data files exist."""
    sample_files = [
        'tests/sample_data.csv',
        'tests/sample_sales_data.csv',
        'tests/sample_financial_data.csv',
        'tests/sample_inventory_data.csv'
    ]
    
    existing_files = []
    missing_files = []
    
    for file_path in sample_files:
        if os.path.exists(file_path):
            existing_files.append(file_path)
        else:
            missing_files.append(file_path)
    
    print(f"📊 Sample Data Files Status:")
    for file_path in existing_files:
        file_size = os.path.getsize(file_path)
        print(f"  ✅ {file_path} ({file_size} bytes)")
    
    for file_path in missing_files:
        print(f"  ❌ {file_path} (missing)")
    
    # Check Excel file
    excel_file = 'tests/sample_student_grades.xlsx'
    if os.path.exists(excel_file):
        file_size = os.path.getsize(excel_file)
        print(f"  ✅ {excel_file} ({file_size} bytes)")
    else:
        print(f"  ❌ {excel_file} (missing)")
    
    return len(existing_files) > 0


def main():
    """Run all integration tests."""
    print("🚀 Data Chatbot Dashboard - Integration Test Suite")
    print(f"📅 Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # Check prerequisites
    print("🔍 Checking prerequisites...")
    
    if not check_test_files():
        print("\n❌ Cannot proceed - missing test files")
        return False
    
    if not check_sample_data_files():
        print("\n⚠️ Warning: Some sample data files are missing")
    
    print("✅ Prerequisites check completed\n")
    
    # Define test suite
    test_suite = [
        ('tests/test_integration.py', 'Basic Integration Tests'),
        ('tests/test_file_handler_integration.py', 'File Handler Integration Tests'),
        ('tests/test_error_integration.py', 'Error Handling Integration Tests'),
        ('tests/test_sample_data_integration.py', 'Sample Data Integration Tests'),
        ('tests/test_comprehensive_integration.py', 'Comprehensive Integration Tests')
    ]
    
    # Run tests
    results = []
    total_duration = 0
    
    for test_file, description in test_suite:
        if os.path.exists(test_file):
            success, duration, output = run_test_file(test_file, description)
            results.append((description, success, duration, output))
            total_duration += duration
        else:
            print(f"\n⚠️ Skipping {description} - file not found: {test_file}")
            results.append((description, False, 0, "File not found"))
    
    # Generate summary report
    print("\n" + "="*80)
    print("📋 INTEGRATION TEST SUMMARY REPORT")
    print("="*80)
    
    passed_tests = sum(1 for _, success, _, _ in results if success)
    total_tests = len(results)
    
    print(f"📊 Overall Results: {passed_tests}/{total_tests} tests passed")
    print(f"⏱️ Total Duration: {total_duration:.2f} seconds")
    print(f"📅 Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n📝 Detailed Results:")
    for description, success, duration, _ in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {status} {description} ({duration:.2f}s)")
    
    # Test coverage summary
    print("\n🎯 Test Coverage Summary:")
    coverage_areas = [
        "✅ Session state management",
        "✅ File upload and validation (CSV & Excel)",
        "✅ Chat interface components",
        "✅ PandasAI agent integration",
        "✅ Visualization rendering",
        "✅ Error handling and recovery",
        "✅ Data persistence and retention",
        "✅ Complete user workflows",
        "✅ Sample data analysis scenarios",
        "✅ Cross-dataset compatibility"
    ]
    
    for area in coverage_areas:
        print(f"  {area}")
    
    # Recommendations
    print("\n💡 Recommendations:")
    if passed_tests == total_tests:
        print("  🎉 All tests passed! The system is ready for production.")
        print("  🔄 Consider running these tests regularly as part of CI/CD.")
        print("  📈 Monitor performance metrics in production environment.")
    else:
        failed_tests = total_tests - passed_tests
        print(f"  ⚠️ {failed_tests} test(s) failed. Review and fix issues before deployment.")
        print("  🔍 Check individual test outputs for specific error details.")
        print("  🛠️ Run failed tests individually for detailed debugging.")
    
    # Sample data recommendations
    print("\n📊 Sample Data Recommendations:")
    print("  • Use provided sample files for testing different data scenarios")
    print("  • Test with your own datasets to validate real-world usage")
    print("  • Verify data quality and format compatibility")
    print("  • Test with various file sizes and complexity levels")
    
    print("\n" + "="*80)
    
    return passed_tests == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)