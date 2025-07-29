#!/usr/bin/env python3

try:
    from ultralytics import YOLO
    import torch
    print("✅ YOLO12 imported successfully!")
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Test model loading
    print("🧪 Testing YOLO12 model loading...")
    model = YOLO('yolo12n.pt')
    print("✅ YOLO12 nano model loaded successfully!")
    print("🚀 YOLO12 is ready for use!")
    
except ImportError as e:
    print(f"⚠️ YOLO12 not available: {e}")
    print("📥 Installing required packages...")
    import subprocess
    import sys
    
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "ultralytics", "torch", "torchvision"])
        print("✅ Packages installed! Please restart the server.")
    except Exception as install_error:
        print(f"❌ Installation failed: {install_error}")
        
except Exception as e:
    print(f"❌ Error: {e}")
