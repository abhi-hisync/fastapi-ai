"""
HISYNC AI API Test Script
Test karne ke liye simple script

© 2025 Hire Synchronisation Pvt. Ltd.
Developer: Abhishek Rajput (@abhi-hisync)
"""

import requests
import json
from pathlib import Path

def test_api_health():
    """API health check karte hain"""
    try:
        print("🏥 Testing API Health...")
        response = requests.get("http://localhost:8000/health")
        if response.status_code == 200:
            print("✅ API is running successfully!")
            print(json.dumps(response.json(), indent=2))
            return True
        else:
            print(f"❌ API health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def test_classification_with_sample():
    """Sample image ke saath classification test"""
    try:
        print("\n🧪 Testing Classification...")
        
        # Create a dummy image file for testing
        from PIL import Image
        import io
        
        # Create a simple coffee-colored image
        img = Image.new('RGB', (224, 224), color=(139, 69, 19))  # Coffee brown color
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        # Test classification endpoint
        files = {'file': ('coffee_test.png', img_byte_arr, 'image/png')}
        data = {'expected_label': 'coffee'}
        
        response = requests.post(
            "http://localhost:8000/classify",
            files=files,
            data=data
        )
        
        if response.status_code == 200:
            print("✅ Classification successful!")
            result = response.json()
            print(f"📊 Status: {result.get('status', 'N/A')}")
            print(f"🏷️ Predicted: {result.get('prediction_result', {}).get('predicted_label', 'N/A')}")
            print(f"📈 Confidence: {result.get('prediction_result', {}).get('confidence', 0) * 100:.1f}%")
            print(f"⚡ Processing Time: {result.get('processing_time_ms', 0):.1f}ms")
            return True
        else:
            print(f"❌ Classification failed: {response.status_code}")
            print(response.text)
            return False
            
    except Exception as e:
        print(f"❌ Error in classification test: {e}")
        return False

def test_stats():
    """API stats check karte hain"""
    try:
        print("\n📊 Getting API Stats...")
        response = requests.get("http://localhost:8000/stats")
        if response.status_code == 200:
            print("✅ Stats retrieved successfully!")
            stats = response.json()
            print(f"🏢 Company: {stats.get('company', 'N/A')}")
            print(f"🤖 Product: {stats.get('product', 'N/A')}")
            print(f"📦 Model Status: {stats.get('model_status', 'N/A')}")
            print(f"🔧 Model Type: {stats.get('model_type', 'N/A')}")
            print(f"📄 API Version: {stats.get('api_version', 'N/A')}")
            return True
        else:
            print(f"❌ Stats check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Error getting stats: {e}")
        return False

def show_api_endpoints():
    """Available endpoints dikhate hain"""
    print("\n🔌 Available API Endpoints:")
    print("="*50)
    print("GET  /                 - Main web interface")
    print("GET  /docs             - Interactive API documentation")
    print("GET  /health           - Health check")
    print("GET  /stats            - API statistics")
    print("GET  /labels           - Supported labels")
    print("GET  /company          - Company information")
    print("POST /classify         - Standard image classification")
    print("POST /classify/resnet  - Google ResNet v2 classification")
    print("POST /classify/batch   - Batch image processing")

def main():
    """Main test function"""
    print("🔥 HISYNC AI - API Test Script")
    print("© 2025 Hire Synchronisation Pvt. Ltd.")
    print("="*50)
    
    # Test 1: Health check
    if not test_api_health():
        print("❌ API is not running. Please start the server first:")
        print("   python main.py")
        return
    
    # Test 2: Stats
    test_stats()
    
    # Test 3: Classification
    test_classification_with_sample()
    
    # Show endpoints
    show_api_endpoints()
    
    print("\n🎉 Testing completed!")
    print("\n📝 How to test manually:")
    print("1. Open browser: http://localhost:8000")
    print("2. Upload image and test classification")
    print("3. Check API docs: http://localhost:8000/docs")
    print("4. Use curl commands:")
    print("""
curl -X POST "http://localhost:8000/classify" \\
     -F "file=@your_image.jpg" \\
     -F "expected_label=coffee"
    """)

if __name__ == "__main__":
    # Install PIL if not available
    try:
        from PIL import Image
    except ImportError:
        print("Installing Pillow...")
        import subprocess
        subprocess.check_call([".venv/Scripts/python.exe", "-m", "pip", "install", "pillow"])
        from PIL import Image
    
    main()
