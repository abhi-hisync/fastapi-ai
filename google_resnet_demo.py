"""
HISYNC AI - Google ResNet v2 Usage Examples
Enhanced coffee classification with superior accuracy

© 2025 Hire Synchronisation Pvt. Ltd.
Developer: Abhishek Rajput (@abhi-hisync)
"""

import requests
import asyncio
import aiohttp
import json
from pathlib import Path

class HISYNCResNetDemo:
    """
    Demo class for Google ResNet v2 coffee classification
    """
    
    def __init__(self, base_url="http://localhost:8000"):
        self.base_url = base_url
        self.endpoints = {
            "standard": f"{base_url}/classify",
            "resnet": f"{base_url}/classify/resnet",
            "health": f"{base_url}/health",
            "stats": f"{base_url}/stats"
        }
    
    def test_health(self):
        """Test API health status"""
        try:
            response = requests.get(self.endpoints["health"])
            print("🏥 Health Check:")
            print(json.dumps(response.json(), indent=2))
            return response.status_code == 200
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
    
    def test_stats(self):
        """Get API statistics"""
        try:
            response = requests.get(self.endpoints["stats"])
            print("📊 API Statistics:")
            print(json.dumps(response.json(), indent=2))
        except Exception as e:
            print(f"❌ Stats request failed: {e}")
    
    def classify_standard(self, image_path, expected_label="coffee"):
        """Test standard classification"""
        try:
            with open(image_path, 'rb') as image_file:
                files = {'file': image_file}
                data = {'expected_label': expected_label}
                
                response = requests.post(self.endpoints["standard"], files=files, data=data)
                
                print("🔍 Standard Classification Result:")
                print(json.dumps(response.json(), indent=2))
                return response.json()
        except Exception as e:
            print(f"❌ Standard classification failed: {e}")
            return None
    
    def classify_resnet(self, image_path, expected_label="coffee"):
        """Test Google ResNet v2 classification"""
        try:
            with open(image_path, 'rb') as image_file:
                files = {'image': image_file}
                data = {
                    'expected_label': expected_label,
                    'confidence_threshold': 0.8
                }
                
                response = requests.post(self.endpoints["resnet"], files=files, data=data)
                
                print("🚀 Google ResNet v2 Classification Result:")
                result = response.json()
                print(json.dumps(result, indent=2))
                
                # Enhanced analysis for ResNet results
                if result.get('coffee_analysis'):
                    print("\n☕ Coffee Analysis Summary:")
                    analysis = result['coffee_analysis']
                    print(f"   Coffee Related: {analysis.get('is_coffee_related', False)}")
                    print(f"   Confidence Level: {analysis.get('confidence_level', 'unknown').upper()}")
                    print(f"   Relevance Score: {analysis.get('confidence_weighted_score', 0) * 100:.1f}%")
                
                if result.get('bluetokie_verification'):
                    print("\n🏢 Bluetokie Verification:")
                    verification = result['bluetokie_verification']
                    print(f"   Recommendation: {verification.get('recommendation', 'N/A')}")
                
                return result
        except Exception as e:
            print(f"❌ ResNet classification failed: {e}")
            return None
    
    def compare_models(self, image_path, expected_label="coffee"):
        """Compare standard vs ResNet v2 performance"""
        print("🔄 Comparing Standard vs Google ResNet v2...")
        print("="*60)
        
        # Test standard model
        print("\n1️⃣ Standard Model:")
        standard_result = self.classify_standard(image_path, expected_label)
        
        print("\n" + "="*60)
        
        # Test ResNet v2 model
        print("\n2️⃣ Google ResNet v2 Model:")
        resnet_result = self.classify_resnet(image_path, expected_label)
        
        print("\n" + "="*60)
        print("\n📈 Performance Comparison:")
        
        if standard_result and resnet_result:
            print(f"Standard Processing Time: {standard_result.get('processing_time_ms', 0):.1f}ms")
            print(f"ResNet v2 Processing Time: {resnet_result.get('processing_time_ms', 0):.1f}ms")
            
            # Extract confidence if available
            std_conf = 0
            if 'prediction_result' in standard_result:
                std_conf = standard_result['prediction_result'].get('confidence', 0)
            
            resnet_conf = 0
            if 'prediction_result' in resnet_result:
                resnet_conf = resnet_result['prediction_result'].get('confidence', 0)
            
            print(f"Standard Confidence: {std_conf * 100:.1f}%")
            print(f"ResNet v2 Confidence: {resnet_conf * 100:.1f}%")
            
            print(f"\n🏆 Recommended Model: {'Google ResNet v2' if resnet_conf > std_conf else 'Standard (for speed)'}")

def main():
    """
    Main demo function
    """
    print("🔥 HISYNC AI - Google ResNet v2 Demo")
    print("© 2025 Hire Synchronisation Pvt. Ltd.")
    print("="*50)
    
    demo = HISYNCResNetDemo()
    
    # Test API health
    if not demo.test_health():
        print("❌ API is not running. Please start the server first:")
        print("   python main.py")
        return
    
    print("\n" + "="*50)
    
    # Get API stats
    demo.test_stats()
    
    print("\n" + "="*50)
    
    # Note about image files
    print("\n📸 Image Testing:")
    print("To test with actual images, place coffee images in the current directory")
    print("Example filenames: coffee.jpg, espresso.png, latte.jpeg")
    
    # Check for sample images
    sample_images = [
        "coffee.jpg", "coffee.png", "coffee.jpeg",
        "espresso.jpg", "espresso.png", "espresso.jpeg",
        "latte.jpg", "latte.png", "latte.jpeg",
        "beans.jpg", "beans.png", "beans.jpeg"
    ]
    
    found_images = []
    for img in sample_images:
        if Path(img).exists():
            found_images.append(img)
    
    if found_images:
        print(f"\n✅ Found sample images: {', '.join(found_images)}")
        
        # Test with first found image
        test_image = found_images[0]
        print(f"\n🧪 Testing with: {test_image}")
        
        # Determine expected label from filename
        expected_label = test_image.split('.')[0]
        
        demo.compare_models(test_image, expected_label)
    else:
        print("\n⚠️ No sample images found. Please add coffee images to test.")
        print("\nExample API usage:")
        print("""
import requests

# Standard classification
with open('coffee.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify',
        files={'file': f},
        data={'expected_label': 'coffee'}
    )

# Google ResNet v2 classification
with open('coffee.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/classify/resnet',
        files={'image': f},
        data={
            'expected_label': 'coffee',
            'confidence_threshold': 0.8
        }
    )
        """)
    
    print("\n🎉 Demo completed!")
    print("🌐 Visit http://localhost:8000 for interactive testing")
    print("📚 Visit http://localhost:8000/docs for API documentation")

if __name__ == "__main__":
    main()
