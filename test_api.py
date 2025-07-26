#!/usr/bin/env python3
"""
Test script for Image Classification API
Tests all endpoints and functionality
"""

import requests
import json
import time
import sys
from pathlib import Path
import io
from PIL import Image
import numpy as np

# API Configuration
BASE_URL = "http://localhost:8000"
HEADERS = {"Content-Type": "application/json"}

def create_test_image(color="red", size=(224, 224)):
    """Create a simple test image"""
    if color == "red":
        rgb = (255, 0, 0)
    elif color == "green":
        rgb = (0, 255, 0)
    elif color == "blue":
        rgb = (0, 0, 255)
    else:
        rgb = (128, 128, 128)
    
    # Create a simple colored image
    img = Image.new('RGB', size, rgb)
    
    # Add some pattern to make it more realistic
    pixels = np.array(img)
    noise = np.random.randint(0, 50, size=pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255)
    
    img = Image.fromarray(pixels.astype(np.uint8))
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def test_health_endpoint():
    """Test the health endpoint"""
    print("🔍 Testing health endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Health check passed: {data}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_root_endpoint():
    """Test the root endpoint"""
    print("🔍 Testing root endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Root endpoint passed: {data['message']}")
            return True
        else:
            print(f"❌ Root endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Root endpoint error: {e}")
        return False

def test_labels_endpoint():
    """Test the labels endpoint"""
    print("🔍 Testing labels endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/labels")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Labels endpoint passed: Found {len(data['supported_labels'])} labels")
            return True
        else:
            print(f"❌ Labels endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Labels endpoint error: {e}")
        return False

def test_stats_endpoint():
    """Test the stats endpoint"""
    print("🔍 Testing stats endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/stats")
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Stats endpoint passed: Model type: {data['model_type']}")
            return True
        else:
            print(f"❌ Stats endpoint failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats endpoint error: {e}")
        return False

def test_classification_endpoint():
    """Test the main classification endpoint"""
    print("🔍 Testing classification endpoint...")
    
    # Create test image
    test_image = create_test_image("red")
    
    try:
        # Test with a generic label
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        data = {
            "expected_label": "car",
            "confidence_threshold": 0.5
        }
        
        response = requests.post(f"{BASE_URL}/classify", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Classification endpoint passed!")
            print(f"   Status: {result['status']}")
            print(f"   Expected: {result['expected_label']}")
            print(f"   Predicted: {result['prediction_result']['predicted_label']}")
            print(f"   Confidence: {result['prediction_result']['confidence']:.2%}")
            print(f"   Processing time: {result['processing_time_ms']:.2f}ms")
            print(f"   Message: {result['message']}")
            return True
        else:
            print(f"❌ Classification endpoint failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Classification endpoint error: {e}")
        return False

def test_invalid_inputs():
    """Test error handling with invalid inputs"""
    print("🔍 Testing error handling...")
    
    # Test 1: Invalid file format
    try:
        files = {"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
        data = {"expected_label": "cat", "confidence_threshold": 0.8}
        
        response = requests.post(f"{BASE_URL}/classify", files=files, data=data)
        
        if response.status_code == 400:
            print("✅ Invalid file format handling works")
        else:
            print(f"❌ Invalid file format should return 400, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
    
    # Test 2: Empty expected label
    try:
        test_image = create_test_image("blue")
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        data = {"expected_label": "", "confidence_threshold": 0.8}
        
        response = requests.post(f"{BASE_URL}/classify", files=files, data=data)
        
        if response.status_code == 400:
            print("✅ Empty label handling works")
        else:
            print(f"❌ Empty label should return 400, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Empty label test failed: {e}")
    
    # Test 3: Invalid confidence threshold
    try:
        test_image = create_test_image("green")
        files = {"image": ("test.jpg", test_image, "image/jpeg")}
        data = {"expected_label": "dog", "confidence_threshold": 1.5}
        
        response = requests.post(f"{BASE_URL}/classify", files=files, data=data)
        
        if response.status_code == 400:
            print("✅ Invalid confidence threshold handling works")
        else:
            print(f"❌ Invalid confidence should return 400, got {response.status_code}")
            
    except Exception as e:
        print(f"❌ Invalid confidence test failed: {e}")

def test_batch_classification():
    """Test batch classification endpoint"""
    print("🔍 Testing batch classification...")
    
    try:
        # Create multiple test images
        test_images = [
            ("test1.jpg", create_test_image("red"), "image/jpeg"),
            ("test2.jpg", create_test_image("blue"), "image/jpeg")
        ]
        
        files = [("images", img) for img in test_images]
        data = {
            "expected_labels": "car,dog",
            "confidence_threshold": 0.5
        }
        
        response = requests.post(f"{BASE_URL}/classify/batch", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"✅ Batch classification passed!")
            print(f"   Total processed: {result['total_processed']}")
            for i, res in enumerate(result['results']):
                print(f"   Image {i+1}: {res['status']} - {res.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ Batch classification failed: {response.status_code}")
            print(f"   Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Batch classification error: {e}")
        return False

def run_performance_test():
    """Run basic performance test"""
    print("🔍 Running performance test...")
    
    processing_times = []
    
    for i in range(5):
        test_image = create_test_image(["red", "green", "blue"][i % 3])
        
        start_time = time.time()
        
        try:
            files = {"image": ("test.jpg", test_image, "image/jpeg")}
            data = {"expected_label": "test", "confidence_threshold": 0.5}
            
            response = requests.post(f"{BASE_URL}/classify", files=files, data=data)
            
            if response.status_code == 200:
                result = response.json()
                processing_times.append(result['processing_time_ms'])
            
        except Exception as e:
            print(f"❌ Performance test iteration {i+1} failed: {e}")
    
    if processing_times:
        avg_time = sum(processing_times) / len(processing_times)
        min_time = min(processing_times)
        max_time = max(processing_times)
        
        print(f"✅ Performance test completed:")
        print(f"   Average processing time: {avg_time:.2f}ms")
        print(f"   Min processing time: {min_time:.2f}ms")
        print(f"   Max processing time: {max_time:.2f}ms")
        print(f"   Processed {len(processing_times)} images")
    else:
        print("❌ Performance test failed - no successful requests")

def main():
    """Run all tests"""
    print("🚀 Starting Image Classification API Tests\n")
    
    # Wait for server to be ready
    print("⏳ Waiting for server to be ready...")
    max_retries = 30
    for i in range(max_retries):
        try:
            response = requests.get(f"{BASE_URL}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is ready!\n")
                break
        except:
            pass
        
        if i == max_retries - 1:
            print("❌ Server not responding. Please start the API first.")
            sys.exit(1)
        
        time.sleep(2)
    
    # Run tests
    tests = [
        ("Root Endpoint", test_root_endpoint),
        ("Health Endpoint", test_health_endpoint),
        ("Labels Endpoint", test_labels_endpoint),
        ("Stats Endpoint", test_stats_endpoint),
        ("Classification Endpoint", test_classification_endpoint),
        ("Error Handling", test_invalid_inputs),
        ("Batch Classification", test_batch_classification),
        ("Performance Test", run_performance_test)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{'='*50}")
        print(f"🧪 {test_name}")
        print('='*50)
        
        try:
            if test_func():
                passed += 1
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"📊 Test Summary")
    print('='*50)
    print(f"✅ Passed: {passed}/{total}")
    print(f"❌ Failed: {total - passed}/{total}")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working perfectly!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please check the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 