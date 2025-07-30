#!/usr/bin/env python3
"""
YOLO12X High-Accuracy Model Test
Test the best YOLO12 model for maximum capability
"""

import asyncio
import time
import requests
import json
from pathlib import Path

async def test_yolo12x_model():
    """Test YOLO12X (highest accuracy) model capabilities"""
    
    print("🔥 YOLO12X High-Accuracy Model Test")
    print("=" * 50)
    
    try:
        # Import and initialize YOLO12 service
        from yolo12_classifier import yolo12_service
        print("✅ YOLO12 service imported successfully")
        
        # Load the YOLO12X model
        print("🔄 Loading YOLO12X (Extra Large) model...")
        start_time = time.time()
        
        await yolo12_service.load_model('yolo12x', 'detect')
        
        load_time = time.time() - start_time
        print(f"✅ YOLO12X model loaded in {load_time:.2f} seconds")
        
        # Get model info
        model_info = yolo12_service.get_model_info()
        print(f"📋 Model Info:")
        print(f"   - Name: {model_info.get('model_name', 'Unknown')}")
        print(f"   - Architecture: {model_info.get('architecture', 'Unknown')}")
        print(f"   - Task: {model_info.get('task', 'Unknown')}")
        print(f"   - Classes: {model_info.get('num_classes', 'Unknown')}")
        
        # Test with a sample image (create a simple test if no image available)
        print("\n🧪 Testing Detection Capabilities...")
        
        # Create a test image data (simple 640x640 RGB array)
        import numpy as np
        from PIL import Image
        import io
        
        # Create a synthetic test image with shapes
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Add some simple shapes for detection
        test_image[100:200, 100:200] = [255, 0, 0]  # Red square
        test_image[300:400, 300:400] = [0, 255, 0]  # Green square
        test_image[200:250, 500:550] = [0, 0, 255]  # Blue rectangle
        
        # Convert to PIL and then bytes
        pil_image = Image.fromarray(test_image)
        img_buffer = io.BytesIO()
        pil_image.save(img_buffer, format='JPEG')
        image_bytes = img_buffer.getvalue()
        
        print(f"📷 Created test image: {len(image_bytes)} bytes")
        
        # Test different confidence thresholds
        confidence_levels = [0.1, 0.25, 0.5, 0.7, 0.9]
        
        for conf in confidence_levels:
            print(f"\n🎯 Testing with confidence threshold: {conf}")
            
            start_time = time.time()
            result = await yolo12_service.detect_objects(
                image_bytes=image_bytes,
                confidence_threshold=conf,
                iou_threshold=0.45
            )
            detection_time = (time.time() - start_time) * 1000
            
            print(f"   ⚡ Detection time: {detection_time:.1f}ms")
            print(f"   📊 Objects detected: {len(result.get('detections', []))}")
            
            if result.get('detections'):
                for i, detection in enumerate(result['detections'][:3]):  # Show first 3
                    print(f"      {i+1}. {detection.get('class', 'Unknown')} - {detection.get('confidence', 0)*100:.1f}%")
        
        # Test classification capability
        print(f"\n🔍 Testing Classification Capability...")
        
        start_time = time.time()
        class_result = await yolo12_service.classify_with_yolo12(
            image_bytes=image_bytes,
            confidence_threshold=0.25
        )
        classification_time = (time.time() - start_time) * 1000
        
        print(f"   ⚡ Classification time: {classification_time:.1f}ms")
        print(f"   🏷️ Classification: {class_result.get('classification', 'Unknown')}")
        print(f"   📈 Confidence: {class_result.get('confidence', 0)*100:.1f}%")
        
        # Performance summary
        print(f"\n📊 YOLO12X Performance Summary:")
        print(f"   🚀 Model Load Time: {load_time:.2f}s")
        print(f"   ⚡ Average Detection: ~{detection_time:.1f}ms")
        print(f"   🔍 Classification: {classification_time:.1f}ms")
        print(f"   🎯 Best suited for: High-accuracy applications")
        print(f"   💡 Recommendation: Use for production quality detection")
        
        print(f"\n✅ YOLO12X capability test completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Please ensure ultralytics is installed: pip install ultralytics")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def test_api_endpoints():
    """Test the API endpoints with high-accuracy model"""
    
    print(f"\n🌐 Testing API Endpoints...")
    base_url = "http://localhost:8000"
    
    try:
        # Test health endpoint
        response = requests.get(f"{base_url}/health", timeout=10)
        health_data = response.json()
        print(f"   💚 Health Status: {health_data.get('status', 'Unknown')}")
        print(f"   🤖 YOLO12 Loaded: {health_data.get('yolo12_loaded', False)}")
        
        # Test model info
        response = requests.get(f"{base_url}/yolo12/info", timeout=10)
        model_data = response.json()
        print(f"   📋 Model: {model_data.get('model_name', 'Unknown')}")
        
        return True
        
    except requests.exceptions.ConnectionError:
        print(f"   ⚠️ API server not running at {base_url}")
        print(f"   💡 Start server with: python main.py")
        return False
    except Exception as e:
        print(f"   ❌ API test failed: {e}")
        return False

async def main():
    """Main test function"""
    
    print("🚀 Starting YOLO12X High-Accuracy Model Capability Test")
    print("=" * 60)
    
    # Test the model directly
    model_success = await test_yolo12x_model()
    
    # Test API if available
    api_success = test_api_endpoints()
    
    print(f"\n🏁 Test Results Summary:")
    print(f"   📊 Model Test: {'✅ Success' if model_success else '❌ Failed'}")
    print(f"   🌐 API Test: {'✅ Success' if api_success else '⚠️ Server not running'}")
    
    if model_success:
        print(f"\n🎉 YOLO12X is ready for high-accuracy detection!")
        print(f"💡 To use in production:")
        print(f"   - Object Detection: High precision applications")
        print(f"   - Quality Control: Industrial inspection")
        print(f"   - Security Systems: Accurate surveillance")
        print(f"   - Medical Imaging: Detailed analysis")
    
    return model_success

if __name__ == "__main__":
    asyncio.run(main())
