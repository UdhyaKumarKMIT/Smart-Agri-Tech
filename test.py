# test1.py
import requests
import json
import os
import webbrowser
import time
from datetime import datetime

# Configuration
API_BASE_URL = "http://localhost:5000"
REPORTS_DIR = "reports"

# Test cases with different soil and climate conditions
TEST_CASES = [
    {
        "name": "Rice Optimal Conditions",
        "description": "High rainfall, warm temperature, balanced nutrients",
        "data": {
            "nitrogen": 80,
            "phosphorus": 45,
            "potassium": 40,
            "temperature": 25.5,
            "humidity": 85,
            "ph": 6.5,
            "rainfall": 220
        }
    },
    {
        "name": "Wheat Ideal Conditions",
        "description": "Moderate rainfall, cooler temperature",
        "data": {
            "nitrogen": 70,
            "phosphorus": 50,
            "potassium": 45,
            "temperature": 18.5,
            "humidity": 65,
            "ph": 7.0,
            "rainfall": 150
        }
    },
    {
        "name": "Maize Growing Conditions",
        "description": "Warm weather, good nitrogen",
        "data": {
            "nitrogen": 90,
            "phosphorus": 40,
            "potassium": 35,
            "temperature": 28.0,
            "humidity": 70,
            "ph": 6.8,
            "rainfall": 180
        }
    },
    {
        "name": "Pulses Ideal",
        "description": "Low nitrogen, moderate rainfall",
        "data": {
            "nitrogen": 30,
            "phosphorus": 55,
            "potassium": 50,
            "temperature": 22.0,
            "humidity": 60,
            "ph": 6.9,
            "rainfall": 120
        }
    },
    {
        "name": "Sugarcane Conditions",
        "description": "High potassium, warm, high rainfall",
        "data": {
            "nitrogen": 75,
            "phosphorus": 50,
            "potassium": 85,
            "temperature": 30.0,
            "humidity": 80,
            "ph": 6.7,
            "rainfall": 200
        }
    },
    {
        "name": "Cotton Growing",
        "description": "Warm, moderate rainfall",
        "data": {
            "nitrogen": 85,
            "phosphorus": 55,
            "potassium": 60,
            "temperature": 27.5,
            "humidity": 65,
            "ph": 7.2,
            "rainfall": 110
        }
    },
    {
        "name": "Extreme Conditions",
        "description": "Very high nutrients, extreme temperatures",
        "data": {
            "nitrogen": 140,
            "phosphorus": 140,
            "potassium": 140,
            "temperature": 35.0,
            "humidity": 90,
            "ph": 8.5,
            "rainfall": 300
        }
    },
    {
        "name": "Poor Soil Conditions",
        "description": "Low nutrients, acidic soil",
        "data": {
            "nitrogen": 20,
            "phosphorus": 15,
            "potassium": 10,
            "temperature": 20.0,
            "humidity": 55,
            "ph": 5.0,
            "rainfall": 80
        }
    }
]

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def print_success(text):
    """Print success message"""
    print(f"✅ {text}")

def print_error(text):
    """Print error message"""
    print(f"❌ {text}")

def print_info(text):
    """Print info message"""
    print(f"📌 {text}")

def print_warning(text):
    """Print warning message"""
    print(f"⚠️ {text}")

def test_server_health():
    """Test if server is running"""
    print_header("TESTING SERVER CONNECTION")
    
    try:
        response = requests.get(f"{API_BASE_URL}/reports/list", timeout=5)
        if response.status_code == 200:
            print_success("Server is running and accessible")
            return True
        else:
            print_error(f"Server returned status code: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Make sure Flask app is running on port 5000")
        print_info("Run 'python app.py' in another terminal first")
        return False
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return False

def test_single_prediction(test_case, with_report=True):
    """Test single prediction with or without report"""
    endpoint = "/predict-crop" if with_report else "/predict-crop-no-report"
    
    print_info(f"Testing: {test_case['name']}")
    print_info(f"Description: {test_case['description']}")
    print_info(f"Endpoint: {endpoint}")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}{endpoint}",
            json=test_case['data'],
            headers={"Content-Type": "application/json"},
            timeout=10
        )
        elapsed_time = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            print_success(f"Prediction successful in {elapsed_time:.2f} seconds")
            print(f"   🌾 Predicted Crop: {result['crop']}")
            print(f"   📊 Confidence: {result['confidence']:.2%}")
            
            if 'top_5_predictions' in result:
                print(f"   📈 Top 5 Predictions:")
                for i, pred in enumerate(result['top_5_predictions'], 1):
                    print(f"      {i}. {pred['crop']}: {pred['probability']:.2%}")
            
            if 'report_path' in result:
                print(f"   📑 Report URL: {API_BASE_URL}{result['report_path']}")
                print(f"   🆔 Report ID: {result['report_id']}")
            
            return result
        else:
            print_error(f"Prediction failed with status {response.status_code}")
            print(f"Response: {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        print_error("Request timed out")
        return None
    except Exception as e:
        print_error(f"Error: {str(e)}")
        return None

def test_batch_predictions():
    """Test multiple predictions in batch"""
    print_header("BATCH PREDICTION TESTING")
    
    results = []
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n--- Test Case {i}/{len(TEST_CASES)} ---")
        result = test_single_prediction(test_case, with_report=True)
        if result:
            results.append({
                "test_case": test_case,
                "result": result
            })
        print("-" * 40)
    
    # Summary
    print_header("BATCH TEST SUMMARY")
    print_success(f"Completed: {len(results)}/{len(TEST_CASES)} tests successful")
    
    # Group predictions by crop
    crop_counts = {}
    for item in results:
        crop = item['result']['crop']
        crop_counts[crop] = crop_counts.get(crop, 0) + 1
    
    if crop_counts:
        print_info("Prediction Distribution:")
        for crop, count in crop_counts.items():
            print(f"   {crop}: {count} times")
    
    return results

def test_without_report():
    """Test prediction without generating report"""
    print_header("TESTING PREDICTION WITHOUT REPORT")
    
    test_case = TEST_CASES[0]  # Use first test case
    result = test_single_prediction(test_case, with_report=False)
    
    if result:
        print_success("Prediction without report successful")
        assert 'report_path' not in result, "Report path should not be present"
        print_info("Verified: No report generated (as expected)")
    
    return result

def test_report_access():
    """Test accessing generated reports"""
    print_header("TESTING REPORT ACCESS")
    
    # First make a prediction to get a report
    test_case = TEST_CASES[0]
    result = test_single_prediction(test_case, with_report=True)
    
    if not result or 'report_id' not in result:
        print_error("Could not get report ID from prediction")
        return False
    
    report_id = result['report_id']
    
    # Test accessing specific report
    try:
        response = requests.get(f"{API_BASE_URL}/report/{report_id}")
        if response.status_code == 200:
            print_success(f"Successfully accessed report: {report_id}")
            print_info(f"Report size: {len(response.content)} bytes")
            print_info(f"Content-Type: {response.headers.get('Content-Type', 'unknown')}")
        else:
            print_error(f"Failed to access report: {response.status_code}")
    except Exception as e:
        print_error(f"Error accessing report: {str(e)}")
    
    # Test accessing latest report
    try:
        response = requests.get(f"{API_BASE_URL}/report/latest")
        if response.status_code == 200:
            print_success("Successfully accessed latest report")
        else:
            print_error(f"Failed to access latest report: {response.status_code}")
    except Exception as e:
        print_error(f"Error accessing latest report: {str(e)}")
    
    return True

def test_reports_list():
    """Test listing all reports"""
    print_header("TESTING REPORTS LIST ENDPOINT")
    
    try:
        response = requests.get(f"{API_BASE_URL}/reports/list")
        if response.status_code == 200:
            data = response.json()
            print_success(f"Found {data['total']} reports")
            
            if data['reports']:
                print_info("Recent reports:")
                for report in data['reports'][:5]:  # Show first 5
                    created = datetime.fromtimestamp(report['created']).strftime('%Y-%m-%d %H:%M:%S')
                    print(f"   📄 {report['id']}")
                    print(f"      Created: {created}")
                    print(f"      URL: {API_BASE_URL}{report['url']}")
            else:
                print_info("No reports found yet")
        else:
            print_error(f"Failed to get reports list: {response.status_code}")
    except Exception as e:
        print_error(f"Error getting reports list: {str(e)}")

def test_invalid_input():
    """Test with invalid input data"""
    print_header("TESTING INVALID INPUT HANDLING")
    
    invalid_cases = [
        {
            "name": "Missing field",
            "data": {
                "nitrogen": 80,
                "phosphorus": 45,
                "potassium": 40,
                "temperature": 25.5,
                "humidity": 85,
                "ph": 6.5
                # missing rainfall
            }
        },
        {
            "name": "Invalid data type",
            "data": {
                "nitrogen": "eighty",  # string instead of number
                "phosphorus": 45,
                "potassium": 40,
                "temperature": 25.5,
                "humidity": 85,
                "ph": 6.5,
                "rainfall": 220
            }
        },
        {
            "name": "Empty data",
            "data": {}
        }
    ]
    
    for case in invalid_cases:
        print_info(f"Testing: {case['name']}")
        try:
            response = requests.post(
                f"{API_BASE_URL}/predict-crop",
                json=case['data'],
                timeout=5
            )
            if response.status_code >= 400:
                print_success(f"Properly handled with status {response.status_code}")
                print(f"   Response: {response.json()}")
            else:
                print_warning(f"Unexpected success with invalid data: {response.status_code}")
        except Exception as e:
            print_error(f"Error: {str(e)}")
        print("-" * 40)

def test_performance():
    """Test API performance"""
    print_header("PERFORMANCE TESTING")
    
    test_case = TEST_CASES[0]
    times = []
    
    print_info("Running 5 consecutive predictions...")
    for i in range(5):
        start_time = time.time()
        response = requests.post(
            f"{API_BASE_URL}/predict-crop",
            json=test_case['data'],
            timeout=10
        )
        elapsed = time.time() - start_time
        times.append(elapsed)
        print(f"   Request {i+1}: {elapsed:.3f} seconds")
    
    avg_time = sum(times) / len(times)
    print_success(f"Average response time: {avg_time:.3f} seconds")
    print_info(f"Min: {min(times):.3f}s, Max: {max(times):.3f}s")

def open_report_in_browser(report_id=None):
    """Open report in browser"""
    if report_id:
        url = f"{API_BASE_URL}/report/{report_id}"
    else:
        url = f"{API_BASE_URL}/report/latest"
    
    print_info(f"Opening {url} in browser...")
    webbrowser.open(url)

def main():
    """Main test function"""
    print_header("🌾 CROP RECOMMENDATION API TEST SUITE")
    print(f"Testing against: {API_BASE_URL}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test server health first
    if not test_server_health():
        print_error("Server not accessible. Exiting tests.")
        print_info("Make sure Flask app is running: python app.py")
        return
    
    # Run all tests
    test_without_report()
    test_single_prediction(TEST_CASES[2])  # Test with specific case
    test_batch_predictions()
    test_report_access()
    test_reports_list()
    test_invalid_input()
    test_performance()
    
    # Summary
    print_header("✅ ALL TESTS COMPLETED")
    print_info("You can view reports at:")
    print(f"   • Latest report: {API_BASE_URL}/report/latest")
    print(f"   • All reports: {API_BASE_URL}/reports/list")
    
    # Ask if user wants to open latest report
    response = input("\n📂 Open latest report in browser? (y/n): ")
    if response.lower() == 'y':
        open_report_in_browser()

def quick_test():
    """Quick test with single prediction"""
    print_header("QUICK TEST - SINGLE PREDICTION")
    
    if not test_server_health():
        return
    
    # Use rice conditions
    test_case = {
        "name": "Quick Test",
        "description": "Rice optimal conditions",
        "data": {
            "nitrogen": 80,
            "phosphorus": 45,
            "potassium": 40,
            "temperature": 25.5,
            "humidity": 85,
            "ph": 6.5,
            "rainfall": 220
        }
    }
    
    result = test_single_prediction(test_case, with_report=True)
    
    if result and 'report_path' in result:
        print_success("Test completed successfully!")
        open_report_in_browser(result['report_id'])

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--quick":
        quick_test()
    else:
        main()