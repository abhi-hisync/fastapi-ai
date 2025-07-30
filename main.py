"""
HISYNC AI - Clean YOLO12 Classification API
Fast and focused object detection and classification using YOLO12

© 2025 Hire Synchronisation Pvt. Ltd.
Developer: Abhishek Rajput (@abhi-hisync)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import asyncio
from contextlib import asynccontextmanager
import time

# Import YOLO12 service
try:
    from yolo12_classifier import yolo12_service
    YOLO12_AVAILABLE = True
    print("✅ YOLO12 Classifier imported successfully")
except ImportError as e:
    YOLO12_AVAILABLE = False
    yolo12_service = None
    print(f"⚠️ YOLO12 Classifier not available: {e}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Application lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    logger.info("🚀 Starting HISYNC AI Clean YOLO12 Application...")
    
    # Initialize YOLO12 service
    if YOLO12_AVAILABLE and yolo12_service:
        logger.info("🔄 Loading YOLO12 model...")
        await yolo12_service.load_model()
        logger.info("✅ YOLO12 model loaded successfully!")
    else:
        logger.warning("⚠️ YOLO12 service not available - running in simulation mode")
    
    yield
    
    logger.info("🛑 Shutting down HISYNC AI Clean YOLO12 Application...")

# FastAPI application
app = FastAPI(
    title="🔥 HISYNC AI - Clean YOLO12 Classification",
    description="Clean and focused YOLO12 object detection and classification API",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
@app.get("/", response_class=HTMLResponse, tags=["🏢 HISYNC General"])
async def read_root():
    """Clean Test Interface for YOLO12"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HISYNC AI - Clean YOLO12 Classification</title>
        <style>
            body { font-family: 'Segoe UI', sans-serif; margin: 0; padding: 20px; 
                   background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: #333; }
            .container { max-width: 1000px; margin: 0 auto; background: white; 
                        border-radius: 15px; padding: 30px; box-shadow: 0 20px 40px rgba(0,0,0,0.1); }
            .header { text-align: center; margin-bottom: 30px; }
            .header h1 { color: #667eea; margin: 0; font-size: 2.5em; }
            .upload-area { border: 3px dashed #667eea; border-radius: 10px; padding: 40px; 
                          text-align: center; margin: 20px 0; background: #f8f9fa; }
            .btn { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; 
                   padding: 15px 30px; border: none; border-radius: 25px; font-size: 16px; 
                   cursor: pointer; margin: 10px; }
            .btn:hover { transform: translateY(-2px); }
            .result-area { margin-top: 30px; padding: 20px; background: #e8f5e8; 
                          border-radius: 10px; display: none; }
            .feature-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); 
                           gap: 15px; margin: 20px 0; }
            .feature-card { background: #f0f8ff; padding: 15px; border-radius: 8px; text-align: center; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 HISYNC AI - Clean YOLO12</h1>
                <p>Clean & Focused Object Detection and Classification</p>
            </div>
            
            <div class="feature-grid">
                <div class="feature-card">
                    <h4>🎯 YOLO12 Detection</h4>
                    <p>State-of-the-art object detection with attention mechanisms</p>
                </div>
                <div class="feature-card">
                    <h4>🔍 Classification</h4>
                    <p>Simple and accurate image classification</p>
                </div>
                <div class="feature-card">
                    <h4>⚡ Fast Processing</h4>
                    <p>Optimized for real-time inference</p>
                </div>
                <div class="feature-card">
                    <h4>🧹 Clean API</h4>
                    <p>Focused endpoints with no bloat</p>
                </div>
            </div>
            
            <div class="upload-area">
                <h3>Upload Image for YOLO12 Analysis</h3>
                <input type="file" id="imageFile" accept="image/*" style="margin: 10px;">
                <br><br>
                <button onclick="detectObjects()" class="btn">🎯 Object Detection</button>
                <button onclick="classifyImage()" class="btn">🔍 Classification</button>
                <button onclick="getModelInfo()" class="btn">📋 Model Info</button>
            </div>
            
            <div class="result-area" id="resultArea">
                <h3>Results</h3>
                <div id="results"></div>
            </div>
        </div>

        <script>
            async function detectObjects() {
                const fileInput = document.getElementById('imageFile');
                if (!fileInput.files[0]) {
                    alert('Please select an image first!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('confidence_threshold', 0.5);
                
                try {
                    const response = await fetch('/yolo12/detect', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    displayResults(data, 'Detection');
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function classifyImage() {
                const fileInput = document.getElementById('imageFile');
                if (!fileInput.files[0]) {
                    alert('Please select an image first!');
                    return;
                }
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('confidence_threshold', 0.5);
                
                try {
                    const response = await fetch('/yolo12/classify', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    displayResults(data, 'Classification');
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            async function getModelInfo() {
                try {
                    const response = await fetch('/yolo12/info');
                    const data = await response.json();
                    displayResults(data, 'Model Info');
                } catch (error) {
                    alert('Error: ' + error.message);
                }
            }
            
            function displayResults(data, type) {
                const results = document.getElementById('results');
                const resultArea = document.getElementById('resultArea');
                
                let html = `<h4>${type} Results</h4>`;
                html += `<p><strong>Status:</strong> ${data.status}</p>`;
                
                if (data.detections) {
                    html += `<p><strong>Objects Detected:</strong> ${data.detections.length}</p>`;
                    data.detections.forEach((detection, i) => {
                        html += `
                            <div style="background: white; padding: 10px; margin: 5px; border-radius: 5px; border-left: 4px solid #667eea;">
                                <strong>${detection.class || detection.class_name || 'Object'}</strong> - 
                                ${((detection.confidence || 0) * 100).toFixed(1)}%
                            </div>
                        `;
                    });
                } else if (data.classification) {
                    html += `
                        <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 8px; border-left: 4px solid #28a745;">
                            <p><strong>Classification:</strong> ${data.classification}</p>
                            <p><strong>Confidence:</strong> ${((data.confidence || 0) * 100).toFixed(1)}%</p>
                        </div>
                    `;
                } else if (data.model_name) {
                    html += `
                        <div style="background: white; padding: 15px; margin: 10px 0; border-radius: 8px;">
                            <p><strong>Model:</strong> ${data.model_name}</p>
                            <p><strong>Available:</strong> ${data.available ? '✅ Yes' : '❌ No'}</p>
                        </div>
                    `;
                }
                
                if (data.processing_time_ms) {
                    html += `<p><strong>Processing Time:</strong> ${data.processing_time_ms.toFixed(0)}ms</p>`;
                }
                
                if (data.message) {
                    html += `<p><em>${data.message}</em></p>`;
                }
                
                results.innerHTML = html;
                resultArea.style.display = 'block';
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/yolo12/detect", tags=["🎯 YOLO12 Detection"])
async def yolo12_detect_objects(
    file: UploadFile = File(..., description="Image file for YOLO12 object detection"),
    confidence_threshold: float = Form(default=0.25, description="Confidence threshold (0.1-1.0)"),
    iou_threshold: float = Form(default=0.45, description="IoU threshold for NMS (0.1-1.0)")
):
    """🎯 YOLO12 Object Detection - Detect objects in images using YOLO12"""
    start_time = time.time()
    
    try:
        # Validate input
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if not 0.1 <= confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.1 and 1.0")
        
        if not 0.1 <= iou_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="IoU threshold must be between 0.1 and 1.0")
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Perform YOLO12 detection
        if YOLO12_AVAILABLE and yolo12_service and yolo12_service.is_loaded:
            result = await yolo12_service.detect_objects(
                image_bytes=image_bytes,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
        else:
            # Simulation mode
            processing_time = (time.time() - start_time) * 1000
            result = {
                "status": "simulation",
                "message": "🔄 YOLO12 Detection in simulation mode",
                "detections": [
                    {"class": "person", "confidence": 0.87, "bbox": {"x1": 100, "y1": 50, "x2": 200, "y2": 300}},
                    {"class": "bicycle", "confidence": 0.73, "bbox": {"x1": 250, "y1": 100, "x2": 400, "y2": 250}}
                ],
                "model_info": {"name": "YOLO12-Simulation", "architecture": "Simulation Mode"},
                "processing_time_ms": processing_time
            }
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        logger.info(f"YOLO12 Detection: {result.get('status')} - {len(result.get('detections', []))} objects")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO12 detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/yolo12/classify", tags=["🔍 YOLO12 Classification"])
async def yolo12_classify_image(
    file: UploadFile = File(..., description="Image file for YOLO12 classification"),
    confidence_threshold: float = Form(default=0.25, description="Confidence threshold"),
    expected_object: str = Form(None, description="Expected object to detect")
):
    """🔍 YOLO12 Image Classification - Classify the main object in an image"""
    start_time = time.time()
    
    try:
        # Validate input
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if not 0.1 <= confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.1 and 1.0")
        
        image_bytes = await file.read()
        
        if YOLO12_AVAILABLE and yolo12_service and yolo12_service.is_loaded:
            result = await yolo12_service.classify_with_yolo12(
                image_bytes=image_bytes,
                expected_object=expected_object,
                confidence_threshold=confidence_threshold
            )
        else:
            # Simulation mode
            processing_time = (time.time() - start_time) * 1000
            result = {
                "status": "simulation",
                "classification": "person",
                "confidence": 0.75,
                "message": "🔄 YOLO12 Classification in simulation mode",
                "processing_time_ms": processing_time
            }
        
        processing_time = (time.time() - start_time) * 1000
        result["processing_time_ms"] = processing_time
        
        logger.info(f"YOLO12 Classification: {result.get('classification')} - {result.get('confidence', 0):.2f}")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO12 classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/yolo12/info", tags=["📋 YOLO12 Information"])
async def get_yolo12_info():
    """Get YOLO12 model information and status"""
    if YOLO12_AVAILABLE and yolo12_service:
        return yolo12_service.get_model_info()
    else:
        return {
            "status": "simulation_mode",
            "model_name": "YOLO12 (Simulation)",
            "available": False,
            "supported_models": {
                'yolo12n': 'nano - fastest, lowest accuracy',
                'yolo12s': 'small - balanced speed/accuracy', 
                'yolo12m': 'medium - good accuracy',
                'yolo12l': 'large - high accuracy',
                'yolo12x': 'extra large - highest accuracy'
            },
            "note": "Install ultralytics package for real YOLO12"
        }

@app.get("/health", tags=["🏢 HISYNC General"])
async def health_check():
    """HISYNC AI Clean YOLO12 system health check"""
    return {
        "status": "healthy" if (YOLO12_AVAILABLE and yolo12_service and yolo12_service.is_loaded) else "simulation",
        "yolo12_loaded": yolo12_service.is_loaded if yolo12_service else False,
        "yolo12_available": YOLO12_AVAILABLE,
        "version": "2.0.0",
        "service": "HISYNC AI - Clean YOLO12 Classification",
        "features": ["YOLO12 Detection", "YOLO12 Classification", "Clean API", "Interactive UI"]
    }

if __name__ == "__main__":
    logger.info("🔥 Starting HISYNC AI Clean YOLO12 Server...")
    uvicorn.run(
        "main:app", 
        host="0.0.0.0", 
        port=8000,
        reload=True,
        log_level="info"
    )
