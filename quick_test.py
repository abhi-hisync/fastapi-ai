import requests
import time
from PIL import Image
import io

# Wait for server to start
print("⏳ Waiting for server to start...")
time.sleep(3)

try:
    # Test health
    print("🏥 Testing Health...")
    response = requests.get("http://localhost:8000/health")
    print(f"Health Status: {response.status_code}")
    if response.status_code == 200:
        print("✅ Server is running!")
        
        # Create test image
        print("\n🧪 Creating test image...")
        img = Image.new('RGB', (224, 224), color=(139, 69, 19))  # Coffee brown
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        # Test ResNet classification
        print("\n🚀 Testing Google ResNet v2...")
        files = {'image': ('test.png', img_bytes, 'image/png')}
        data = {
            'expected_label': 'coffee',
            'confidence_threshold': 0.8
        }
        
        response = requests.post(
            "http://localhost:8000/classify/resnet",
            files=files,
            data=data
        )
        
        print(f"ResNet Response Status: {response.status_code}")
        print(f"ResNet Response: {response.text[:500]}...")
        
        if response.status_code == 200:
            print("✅ Google ResNet v2 working!")
        else:
            print("❌ ResNet failed")
            
    else:
        print("❌ Server not responding")
        
except Exception as e:
    print(f"❌ Error: {e}")
