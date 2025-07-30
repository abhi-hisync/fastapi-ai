# 🔥 HISYNC AI - Clean YOLO12 Classification API

A clean, focused FastAPI application for object detection and classification using YOLO12.

## ✅ Features

- **YOLO12 Object Detection**: State-of-the-art attention-centric object detection
- **Image Classification**: Simple classification endpoint
- **Clean API**: Minimal, focused endpoints
- **Health Monitoring**: Built-in health checks
- **Easy Setup**: Single command deployment
- **Interactive UI**: Built-in web interface for testing

## 🚀 Quick Start

### Option 1: Auto Start
```bash
# Install dependencies and start server
pip install -r requirements.txt
python main.py
```

### Option 2: Using uvicorn directly
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## 📡 API Endpoints

### 1. Root Interface
```
GET /
```
Interactive web interface for testing YOLO12 detection and classification.

### 2. Health Check
```
GET /health
```
Returns system health status and YOLO12 model information.

### 3. Object Detection
```
POST /yolo12/detect
- file: image file
- confidence_threshold: float (default: 0.25)
- iou_threshold: float (default: 0.45)
```

### 4. Image Classification
```
POST /yolo12/classify
- file: image file
- confidence_threshold: float (default: 0.25)
- expected_object: string (optional)
```

### 5. Model Information
```
GET /yolo12/info
```
Get YOLO12 model details and supported model variants.

## 🧪 Testing

### Web Interface
1. Start the server: `python main.py`
2. Open your browser to `http://localhost:8000`
3. Upload an image and click "YOLO12 Detection" or "Classification"

### API Testing
```bash
# Test detection
curl -X POST "http://localhost:8000/yolo12/detect" \
     -F "file=@your_image.jpg" \
     -F "confidence_threshold=0.5"

# Test classification
curl -X POST "http://localhost:8000/yolo12/classify" \
     -F "file=@your_image.jpg" \
     -F "confidence_threshold=0.5"
```

### Python Test Script
```bash
python test_yolo12.py
```

## 📁 Clean Project Structure

```
fastapi-ai/
├── main.py                    # Main FastAPI application (clean & focused)
├── yolo12_classifier.py       # YOLO12 service implementation
├── main_yolo12.py            # Alternative YOLO12 server
├── yolo12_only_server.py     # Minimal YOLO12 server
├── test_yolo12.py            # YOLO12 test script
├── requirements.txt          # Clean dependencies
├── models.py                 # Pydantic models
├── yolo12n.pt               # YOLO12 nano model
├── YOLO12_INTEGRATION_COMPLETE.md  # YOLO12 documentation
└── README.md                 # This file
```

## 🔧 Configuration

The API automatically:
- Downloads YOLO12 model on first run if not present
- Serves on http://localhost:8000
- Provides interactive docs at `/docs` and `/redoc`
- Supports CORS for web applications

## 📊 Response Formats

### Detection Response
```json
{
  "status": "success",
  "message": "✅ YOLO12 detected 3 objects successfully!",
  "detections": [
    {
      "class": "person",
      "confidence": 0.85,
      "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}
    }
  ],
  "model_info": {
    "name": "YOLO12-NANO",
    "architecture": "Attention-Centric with Area Attention & R-ELAN"
  },
  "processing_time_ms": 45.2
}
```

### Classification Response
```json
{
  "status": "success",
  "classification": "person",
  "confidence": 0.85,
  "message": "Successfully classified image",
  "processing_time_ms": 32.1
}
```

## 🎯 What's Been Cleaned

### ❌ Removed (Unnecessary Files):
- ✅ All Bluetokie training files (`bluetokie_dataset_*.py`)
- ✅ Google integrations (`google_*.py`)
- ✅ ResNet implementations (`resnet_*.py`)
- ✅ Coffee-specific classifiers (`coffee_classifier.py`, `main_coffee.py`)
- ✅ Training data directories (`bluetokie_training_data/`)
- ✅ Setup and deployment scripts
- ✅ Legacy test files and examples

### ✅ Kept (Essential Components):
- 🎯 YOLO12 core implementation (`yolo12_classifier.py`)
- 🎯 Clean main API (`main.py`)
- 🎯 YOLO12 model file (`yolo12n.pt`)
- 🎯 Essential utilities (`models.py`, `test_yolo12.py`)
- 🎯 Documentation (`YOLO12_INTEGRATION_COMPLETE.md`)

## 🚀 Ready to Use

This project is now **clean, manageable, and focused** on:
- ✅ YOLO12 object detection
- ✅ Simple classification
- ✅ Clean API design
- ✅ Easy deployment

No unnecessary Google integrations, complex training pipelines, or redundant files.

## 📞 Support

- **Developer**: Abhishek Rajput (@abhi-hisync)
- **Company**: Hire Synchronisation Pvt. Ltd.
- **Email**: support@hisync.in
- **GitHub**: https://github.com/abhi-hisync/fastapi-ai

---

**Ready to use YOLO12 for object detection!** 🚀
