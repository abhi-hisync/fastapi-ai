#!/usr/bin/env python3
"""
Example usage of the Image Classification API
Demonstrates how to use the API for audit automation
"""

import requests
import json
from PIL import Image
import io
import numpy as np

# API Configuration
API_BASE_URL = "http://localhost:8000"

def create_sample_image(image_type="cat"):
    """Create a sample image for testing"""
    # Create a simple test image
    if image_type == "cat":
        # Create an orange/brown colored image (cat-like colors)
        color = (200, 120, 60)
    elif image_type == "dog":
        # Create a brown colored image (dog-like colors)
        color = (139, 69, 19)
    elif image_type == "car":
        # Create a blue colored image (car-like colors)
        color = (70, 130, 180)
    else:
        color = (128, 128, 128)
    
    # Create image with some texture
    img = Image.new('RGB', (224, 224), color)
    pixels = np.array(img)
    
    # Add some noise/texture to make it more realistic
    noise = np.random.randint(-30, 30, size=pixels.shape)
    pixels = np.clip(pixels + noise, 0, 255)
    
    # Add some geometric patterns
    for i in range(0, 224, 20):
        for j in range(0, 224, 20):
            if (i + j) % 40 == 0:
                pixels[i:i+10, j:j+10] = [min(255, c + 50) for c in color]
    
    img = Image.fromarray(pixels.astype(np.uint8))
    
    # Convert to bytes
    img_bytes = io.BytesIO()
    img.save(img_bytes, format='JPEG')
    img_bytes.seek(0)
    
    return img_bytes

def classify_image(image_bytes, expected_label, confidence_threshold=0.8):
    """Classify an image using the API"""
    print(f"\n🔍 Classifying image with expected label: '{expected_label}'")
    print(f"📊 Confidence threshold: {confidence_threshold}")
    
    try:
        # Prepare the request
        files = {"image": ("test_image.jpg", image_bytes, "image/jpeg")}
        data = {
            "expected_label": expected_label,
            "confidence_threshold": confidence_threshold
        }
        
        # Make the API request
        response = requests.post(f"{API_BASE_URL}/classify", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            print(f"\n📋 Classification Results:")
            print(f"   Status: {result['status'].upper()}")
            print(f"   Expected: {result['expected_label']}")
            print(f"   Predicted: {result['prediction_result']['predicted_label']}")
            print(f"   Confidence: {result['prediction_result']['confidence']:.2%}")
            print(f"   Match: {'✅ Yes' if result['is_match'] else '❌ No'}")
            print(f"   Confidence Met: {'✅ Yes' if result['confidence_met'] else '❌ No'}")
            print(f"   Processing Time: {result['processing_time_ms']:.2f}ms")
            print(f"\n💬 Message: {result['message']}")
            
            # Show top predictions
            print(f"\n🏆 Top Predictions:")
            for i, pred in enumerate(result['prediction_result']['all_predictions'][:3]):
                print(f"   {i+1}. {pred['label']} ({pred['confidence']:.2%})")
            
            return result
            
        else:
            print(f"❌ API Error: {response.status_code}")
            print(f"   Response: {response.text}")
            return None
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

def batch_classify_images():
    """Demonstrate batch classification"""
    print(f"\n🔍 Batch Classification Demo")
    print("="*50)
    
    try:
        # Create multiple test images
        cat_image = create_sample_image("cat")
        dog_image = create_sample_image("dog")
        
        # Prepare batch request
        files = [
            ("images", ("cat.jpg", cat_image, "image/jpeg")),
            ("images", ("dog.jpg", dog_image, "image/jpeg"))
        ]
        data = {
            "expected_labels": "cat,dog",
            "confidence_threshold": 0.7
        }
        
        response = requests.post(f"{API_BASE_URL}/classify/batch", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            
            print(f"📦 Batch Results (Total: {result['total_processed']})")
            for i, res in enumerate(result['results']):
                print(f"\n   Image {i+1} ({res.get('filename', 'N/A')}):")
                print(f"   Status: {res['status'].upper()}")
                if 'prediction_result' in res:
                    print(f"   Predicted: {res['prediction_result']['predicted_label']}")
                    print(f"   Confidence: {res['prediction_result']['confidence']:.2%}")
                print(f"   Message: {res.get('message', 'N/A')}")
        else:
            print(f"❌ Batch API Error: {response.status_code}")
            
    except Exception as e:
        print(f"❌ Batch Error: {e}")

def check_api_health():
    """Check if the API is healthy"""
    print("🏥 Checking API Health...")
    
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API Status: {health['status']}")
            print(f"🤖 Model Loaded: {health['model_loaded']}")
            print(f"📋 Supported Formats: {', '.join(health['supported_formats'])}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def main():
    """Main demonstration function"""
    print("🚀 Image Classification API Demo")
    print("="*50)
    print("This demonstrates audit automation using image classification")
    
    # Check API health first
    if not check_api_health():
        print("\n❌ API is not available. Please start the server first:")
        print("   python main.py")
        return
    
    print("\n" + "="*50)
    print("📸 Single Image Classification Examples")
    print("="*50)
    
    # Example 1: Correct classification (high confidence)
    print("\n🎯 Example 1: Expected CORRECT result")
    cat_image = create_sample_image("cat")
    classify_image(cat_image, "cat", confidence_threshold=0.5)
    
    # Example 2: Different expected vs predicted
    print("\n🎯 Example 2: Expected INCORRECT result")
    cat_image = create_sample_image("cat")
    classify_image(cat_image, "dog", confidence_threshold=0.8)
    
    # Example 3: Low confidence scenario
    print("\n🎯 Example 3: Low confidence scenario")
    test_image = create_sample_image("car")
    classify_image(test_image, "car", confidence_threshold=0.9)
    
    # Batch classification demo
    print("\n" + "="*50)
    print("📦 Batch Classification Demo")
    print("="*50)
    batch_classify_images()
    
    # Summary
    print("\n" + "="*50)
    print("🎉 Demo Complete!")
    print("="*50)
    print("✅ Your audit automation system is ready!")
    print("📋 Key Features Demonstrated:")
    print("   • Single image classification")
    print("   • Batch processing")
    print("   • Confidence scoring")
    print("   • Match verification")
    print("   • Error handling")
    print("\n🌐 Interactive Documentation: http://localhost:8000/docs")
    print("📊 API Health: http://localhost:8000/health")

if __name__ == "__main__":
    main() 