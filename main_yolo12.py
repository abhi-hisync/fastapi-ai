"""
HISYNC AI - YOLO12 Main Application
FastAPI application with YOLO12 attention-centric object detection
Specialized for comprehensive object detection and coffee/cafe verification

© 2025 Hire Synchronisation Pvt. Ltd.
Developer: Abhishek Rajput (@abhi-hisync)
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import logging
import asyncio
from contextlib import asynccontextmanager

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

# Pydantic models for API responses
class YOLODetection(BaseModel):
    id: int
    class_name: str = Field(alias='class')
    class_id: int
    confidence: float
    is_coffee_related: bool
    adjusted_confidence: float
    bbox: Dict[str, float]

class YOLOResponse(BaseModel):
    status: str
    message: str
    model_info: Dict[str, Any]
    detections: List[Dict[str, Any]]
    detection_summary: Dict[str, Any]
    coffee_analysis: Dict[str, Any]
    bluetokie_verification: Dict[str, Any]
    processing_info: Dict[str, Any]
    performance_stats: Optional[Dict[str, Any]] = None

# Startup and shutdown events
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup and shutdown events"""
    # Startup
    logger.info("🚀 Starting HISYNC AI YOLO12 Detection API...")
    try:
        if YOLO12_AVAILABLE and yolo12_service:
            await yolo12_service.load_model('yolo12n', 'detect')  # Start with nano for speed
            logger.info("✅ YOLO12 Attention-Centric Object Detector loaded successfully!")
        else:
            logger.warning("⚠️ YOLO12 not available, API will run in simulation mode")
        
    except Exception as e:
        logger.error(f"❌ Failed to load YOLO12 models: {e}")
        # Don't raise error, let it run with fallback
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down HISYNC AI YOLO12 Detection API...")

# Create FastAPI instance
app = FastAPI(
    title="🔥 HISYNC AI - YOLO12 Attention-Centric Object Detection API",
    description="""
    **HISYNC AI - YOLO12 Attention-Centric Object Detection API** 
    
    🚀 **NEW: YOLO12 Integration** - The latest attention-centric object detection with superior accuracy!
    
    Powered by **Hire Synchronisation Pvt. Ltd.** - Your trusted partner in AI-driven business solutions.
    
    ## 👨‍💻 Developer
    **Developed by**: Abhishek Rajput  
    **GitHub**: [@abhi-hisync](https://github.com/abhi-hisync)  
    **Project Repository**: [fastapi-ai](https://github.com/abhi-hisync/fastapi-ai)
    
    ## 🤖 YOLO12 Features (Attention-Centric Architecture)
    - 🎯 **Area Attention Mechanism**: Efficient large receptive field processing
    - 🔍 **R-ELAN**: Residual Efficient Layer Aggregation Networks  
    - 📊 **FlashAttention**: Optimized attention architecture for speed
    - 🛡️ **Multi-Task Support**: Detection, Segmentation, Classification, Pose, OBB
    - 📈 **State-of-the-Art**: Superior accuracy with fewer parameters
    - 🔒 **Enterprise Security**: Advanced input validation and secure file handling
    - ☕ **Coffee Specialized**: Enhanced for Bluetokie coffee/cafe detection
    
    ## 🌟 Perfect for Enterprise Use
    - **Object Detection**: Comprehensive real-time object identification
    - **Quality Assurance**: Intelligent quality control processes
    - **Asset Management**: Smart asset identification and tracking
    - **Cafe Verification**: Specialized coffee shop and cafe environment detection
    - **Audit Automation**: Automated verification for Bluetokie standards
    
    ## 🚀 Why Choose HISYNC AI YOLO12?
    ✅ **Real-Time Performance** - Attention-optimized inference  
    ✅ **State-of-the-Art Accuracy** - Latest YOLO12 attention mechanisms
    ✅ **Multi-Task Capable** - Detection, segmentation, classification in one model
    ✅ **Scalable** - Handle multiple concurrent requests  
    ✅ **Secure** - Bank-level security standards  
    ✅ **Coffee Optimized** - Specialized for Bluetokie verification workflows
    
    ---
    **© 2025 Hire Synchronisation Pvt. Ltd. All rights reserved.**  
    **Developer**: Abhishek Rajput ([@abhi-hisync](https://github.com/abhi-hisync))  
    **Contact**: support@hisync.in | **Website**: https://hisync.in  
    **Source Code**: https://github.com/abhi-hisync/fastapi-ai
    """,
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    contact={
        "name": "Hire Synchronisation Pvt. Ltd.",
        "url": "https://hisync.in",
        "email": "support@hisync.in"
    },
    license_info={
        "name": "HISYNC Proprietary License",
        "url": "https://hisync.in/license"
    }
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint with enhanced testing interface
@app.get("/", response_class=HTMLResponse, tags=["🏢 HISYNC General"])
async def read_root():
    """Enhanced Test Interface for YOLO12 Object Detection"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>HISYNC AI - YOLO12 Object Detection Test Interface</title>
        <style>
            body {
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                margin: 0;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: #333;
            }
            .container {
                max-width: 1400px;
                margin: 0 auto;
                background: white;
                border-radius: 15px;
                padding: 30px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
            }
            .header {
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 3px solid #667eea;
            }
            .header h1 {
                color: #667eea;
                margin: 0;
                font-size: 2.5em;
                text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
            }
            .yolo12-badge {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
                color: white;
                padding: 8px 16px;
                border-radius: 20px;
                font-weight: bold;
                display: inline-block;
                margin: 10px 0;
            }
            .test-section {
                background: #f8f9fa;
                padding: 25px;
                border-radius: 10px;
                margin: 20px 0;
                border-left: 5px solid #667eea;
            }
            .model-selector {
                margin: 20px 0;
                padding: 20px;
                background: #e3f2fd;
                border-radius: 10px;
            }
            .model-selector select, .model-selector input {
                margin: 10px;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
            }
            .upload-area {
                border: 3px dashed #667eea;
                border-radius: 10px;
                padding: 40px;
                text-align: center;
                margin: 20px 0;
                background: #f8f9fa;
                transition: all 0.3s ease;
            }
            .upload-area:hover {
                background: #e9ecef;
                border-color: #764ba2;
            }
            .file-input {
                margin: 20px 0;
                padding: 10px;
                border: 2px solid #ddd;
                border-radius: 5px;
                font-size: 16px;
                width: 100%;
                box-sizing: border-box;
            }
            .btn {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 15px 30px;
                border: none;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                margin: 5px;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 8px 25px rgba(102, 126, 234, 0.6);
            }
            .btn-yolo12 {
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%);
            }
            .result-area {
                margin-top: 30px;
                padding: 20px;
                background: #e8f5e8;
                border-radius: 10px;
                border-left: 5px solid #28a745;
                display: none;
            }
            .detection-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 15px;
                margin: 20px 0;
            }
            .detection-card {
                background: white;
                padding: 15px;
                border-radius: 8px;
                border: 1px solid #ddd;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }
            .confidence-bar {
                background: #ddd;
                height: 8px;
                border-radius: 4px;
                margin: 5px 0;
                overflow: hidden;
            }
            .confidence-fill {
                height: 100%;
                background: linear-gradient(90deg, #28a745, #20c997);
                transition: width 0.3s ease;
            }
            .features-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 30px 0;
            }
            .feature-card {
                background: white;
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                border-top: 4px solid #ff6b6b;
            }
            .loading {
                text-align: center;
                margin: 20px 0;
                display: none;
            }
            .loading::after {
                content: '';
                display: inline-block;
                width: 30px;
                height: 30px;
                border: 3px solid #f3f3f3;
                border-radius: 50%;
                border-top-color: #ff6b6b;
                animation: spin 1s ease-in-out infinite;
            }
            @keyframes spin {
                to { transform: rotate(360deg); }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔥 HISYNC AI - YOLO12 Object Detection</h1>
                <div class="yolo12-badge">YOLO12 Attention-Centric Architecture</div>
                <div>State-of-the-Art Object Detection with Area Attention & R-ELAN</div>
            </div>

            <div class="test-section">
                <h2>🧪 Live YOLO12 Object Detection Test</h2>
                
                <div class="model-selector">
                    <h3>🎯 Model Configuration</h3>
                    <label>Model Size:</label>
                    <select id="modelSize">
                        <option value="yolo12n">YOLO12n - Nano (Fastest)</option>
                        <option value="yolo12s">YOLO12s - Small (Balanced)</option>
                        <option value="yolo12m">YOLO12m - Medium (Good Accuracy)</option>
                        <option value="yolo12l">YOLO12l - Large (High Accuracy)</option>
                        <option value="yolo12x">YOLO12x - Extra Large (Best Accuracy)</option>
                    </select>
                    
                    <label>Confidence Threshold:</label>
                    <input type="range" id="confidenceSlider" min="0.1" max="1.0" step="0.05" value="0.25">
                    <span id="confidenceValue">0.25</span>
                    
                    <label>IoU Threshold:</label>
                    <input type="range" id="iouSlider" min="0.1" max="1.0" step="0.05" value="0.45">
                    <span id="iouValue">0.45</span>
                </div>
                
                <div class="upload-area">
                    <h3>Upload Image for YOLO12 Detection</h3>
                    <p>Select an image to test our advanced YOLO12 object detection system</p>
                    <input type="file" id="imageFile" class="file-input" accept="image/*">
                    <br><br>
                    <button onclick="detectObjects()" class="btn btn-yolo12">🎯 YOLO12 Object Detection</button>
                    <button onclick="switchModel()" class="btn">🔄 Switch Model</button>
                </div>
                
                <div class="loading" id="loading">
                    <p>🤖 YOLO12 is analyzing your image...</p>
                </div>
                
                <div class="result-area" id="resultArea">
                    <h3>📊 YOLO12 Detection Results</h3>
                    <div id="results"></div>
                </div>
            </div>

            <div class="features-grid">
                <div class="feature-card">
                    <h3>🎯 Area Attention Mechanism</h3>
                    <p>Revolutionary attention approach that processes large receptive fields efficiently by dividing feature maps into regions</p>
                </div>
                <div class="feature-card">
                    <h3>🔗 R-ELAN Architecture</h3>
                    <p>Residual Efficient Layer Aggregation Networks with block-level residual connections and scaling</p>
                </div>
                <div class="feature-card">
                    <h3>⚡ FlashAttention Integration</h3>
                    <p>Optimized attention mechanism that minimizes memory access overhead for faster inference</p>
                </div>
                <div class="feature-card">
                    <h3>☕ Coffee Environment Detection</h3>
                    <p>Specialized analysis for coffee shops, cafes, and Bluetokie verification workflows</p>
                </div>
            </div>

            <div class="test-section">
                <h2>📋 YOLO12 Capabilities</h2>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                    <div style="background: #fff3cd; padding: 15px; border-radius: 8px; text-align: center;">
                        <strong>Object Detection</strong><br>80+ COCO classes
                    </div>
                    <div style="background: #d1ecf1; padding: 15px; border-radius: 8px; text-align: center;">
                        <strong>Instance Segmentation</strong><br>Pixel-level precision
                    </div>
                    <div style="background: #d4edda; padding: 15px; border-radius: 8px; text-align: center;">
                        <strong>Classification</strong><br>Image categorization
                    </div>
                    <div style="background: #f8d7da; padding: 15px; border-radius: 8px; text-align: center;">
                        <strong>Pose Estimation</strong><br>Human pose detection
                    </div>
                    <div style="background: #e2e3e5; padding: 15px; border-radius: 8px; text-align: center;">
                        <strong>Oriented Detection</strong><br>Rotated bounding boxes
                    </div>
                </div>
            </div>
        </div>

        <script>
            // Update slider values
            document.getElementById('confidenceSlider').oninput = function() {
                document.getElementById('confidenceValue').textContent = this.value;
            }
            document.getElementById('iouSlider').oninput = function() {
                document.getElementById('iouValue').textContent = this.value;
            }
            
            async function detectObjects() {
                const fileInput = document.getElementById('imageFile');
                const loading = document.getElementById('loading');
                const resultArea = document.getElementById('resultArea');
                const results = document.getElementById('results');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image file first!');
                    return;
                }
                
                // Show loading
                loading.style.display = 'block';
                resultArea.style.display = 'none';
                
                const formData = new FormData();
                formData.append('file', fileInput.files[0]);
                formData.append('confidence_threshold', document.getElementById('confidenceSlider').value);
                formData.append('iou_threshold', document.getElementById('iouSlider').value);
                
                try {
                    const response = await fetch('/yolo12/detect', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        displayYOLOResults(data);
                        resultArea.style.display = 'block';
                    } else {
                        throw new Error(data.detail || 'Detection failed');
                    }
                } catch (error) {
                    results.innerHTML = `<div style="color: red; padding: 20px;"><strong>Error:</strong> ${error.message}</div>`;
                    resultArea.style.display = 'block';
                } finally {
                    loading.style.display = 'none';
                }
            }
            
            function displayYOLOResults(data) {
                const results = document.getElementById('results');
                
                let html = `
                    <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ee5a52 100%); color: white; padding: 15px; border-radius: 10px; margin: 10px 0;">
                        <h4>🎯 YOLO12 Detection Results</h4>
                        <p><strong>Status:</strong> ${data.status}</p>
                        <p><strong>Model:</strong> ${data.model_info?.name || 'YOLO12'}</p>
                        <p><strong>Objects Detected:</strong> ${data.detection_summary?.total_objects || 0}</p>
                    </div>
                `;
                
                if (data.detections && data.detections.length > 0) {
                    html += '<div class="detection-grid">';
                    
                    data.detections.forEach(detection => {
                        const confidencePercent = (detection.confidence * 100).toFixed(1);
                        const coffeeIcon = detection.is_coffee_related ? '☕' : '📦';
                        
                        html += `
                            <div class="detection-card">
                                <h4>${coffeeIcon} ${detection.class}</h4>
                                <p><strong>Confidence:</strong> ${confidencePercent}%</p>
                                <div class="confidence-bar">
                                    <div class="confidence-fill" style="width: ${confidencePercent}%"></div>
                                </div>
                                <p><strong>Coffee Related:</strong> ${detection.is_coffee_related ? '✅ Yes' : '❌ No'}</p>
                                <p><strong>Box:</strong> ${Math.round(detection.bbox.width)}×${Math.round(detection.bbox.height)}</p>
                            </div>
                        `;
                    });
                    
                    html += '</div>';
                } else {
                    html += '<p>No objects detected. Try adjusting the confidence threshold or using a different image.</p>';
                }
                
                // Add coffee analysis
                if (data.coffee_analysis) {
                    html += `
                        <div style="background: white; padding: 20px; border-radius: 10px; margin: 15px 0; border: 2px solid #28a745;">
                            <h4>☕ Coffee Environment Analysis</h4>
                            <p><strong>Cafe Environment:</strong> ${data.coffee_analysis.is_cafe_environment ? '✅ Detected' : '❌ Not Detected'}</p>
                            <p><strong>Coffee Context Score:</strong> ${(data.coffee_analysis.coffee_context_score * 100).toFixed(1)}%</p>
                            <p><strong>Confidence Level:</strong> ${data.coffee_analysis.confidence_level?.toUpperCase() || 'N/A'}</p>
                            ${data.coffee_analysis.detected_coffee_items?.length > 0 ? `
                                <p><strong>Coffee Items:</strong> ${data.coffee_analysis.detected_coffee_items.join(', ')}</p>
                            ` : ''}
                        </div>
                    `;
                }
                
                // Add Bluetokie verification
                if (data.bluetokie_verification) {
                    html += `
                        <div style="background: white; padding: 20px; border-radius: 10px; margin: 15px 0; border: 2px solid #007bff;">
                            <h4>🏢 Bluetokie Verification</h4>
                            <p><strong>Recommendation:</strong> ${data.bluetokie_verification.recommendation}</p>
                            <p><strong>Audit Score:</strong> ${(data.bluetokie_verification.audit_score * 100).toFixed(1)}%</p>
                        </div>
                    `;
                }
                
                // Add processing info
                if (data.processing_info) {
                    html += `
                        <div style="background: white; padding: 20px; border-radius: 10px; margin: 15px 0;">
                            <h4>⚡ Processing Information</h4>
                            <p><strong>Processing Time:</strong> ${data.processing_info.processing_time_ms?.toFixed(0) || 'N/A'} ms</p>
                            <p><strong>YOLO12 Features:</strong> ${data.processing_info.yolo12_features_used?.join(', ') || 'Area Attention, R-ELAN'}</p>
                        </div>
                    `;
                }
                
                results.innerHTML = html;
            }
            
            async function switchModel() {
                const modelSize = document.getElementById('modelSize').value;
                
                try {
                    const response = await fetch('/yolo12/switch-model', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify({
                            model_size: modelSize,
                            task: 'detect'
                        })
                    });
                    
                    const data = await response.json();
                    
                    if (response.ok) {
                        alert(`✅ Switched to ${modelSize.toUpperCase()} successfully!`);
                    } else {
                        alert(`❌ Failed to switch model: ${data.detail}`);
                    }
                } catch (error) {
                    alert(`❌ Error: ${error.message}`);
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

# YOLO12 object detection endpoint
@app.post("/yolo12/detect", tags=["🎯 YOLO12 Detection"])
async def yolo12_detect_objects(
    file: UploadFile = File(..., description="Image file for YOLO12 object detection"),
    confidence_threshold: float = Form(default=0.25, description="Confidence threshold (0.1-1.0)"),
    iou_threshold: float = Form(default=0.45, description="IoU threshold for NMS (0.1-1.0)")
):
    """
    🎯 **YOLO12 Attention-Centric Object Detection**
    
    Perform state-of-the-art object detection using YOLO12's revolutionary attention mechanisms.
    
    **YOLO12 Features:**
    - 🎯 Area Attention Mechanism for efficient large receptive field processing
    - 🔗 R-ELAN (Residual Efficient Layer Aggregation Networks)
    - ⚡ FlashAttention optimization for reduced memory overhead
    - 📊 Superior accuracy with fewer parameters
    - ☕ Enhanced coffee/cafe environment detection
    """
    try:
        if not YOLO12_AVAILABLE or not yolo12_service:
            raise HTTPException(
                status_code=503,
                detail="YOLO12 service not available. Please check installation."
            )
        
        # Validate file type
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=400,
                detail="Invalid file type. YOLO12 requires valid image files."
            )
        
        # Validate thresholds
        if not 0.1 <= confidence_threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="Confidence threshold must be between 0.1 and 1.0"
            )
        
        if not 0.1 <= iou_threshold <= 1.0:
            raise HTTPException(
                status_code=400,
                detail="IoU threshold must be between 0.1 and 1.0"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Perform YOLO12 detection
        result = await yolo12_service.detect_objects(
            image_bytes=image_bytes,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold
        )
        
        logger.info(f"YOLO12 Detection: {result.get('status', 'completed')} - {len(result.get('detections', []))} objects detected")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO12 detection error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"YOLO12 detection failed: {str(e)}"
        )

# Model switching endpoint
@app.post("/yolo12/switch-model", tags=["🎯 YOLO12 Detection"])
async def switch_yolo12_model(
    model_size: str = Form(..., description="Model size (yolo12n, yolo12s, yolo12m, yolo12l, yolo12x)"),
    task: str = Form(default="detect", description="Task type (detect, segment, classify, pose, obb)")
):
    """
    🔄 **Switch YOLO12 Model**
    
    Dynamically switch between different YOLO12 model sizes and tasks.
    """
    try:
        if not YOLO12_AVAILABLE or not yolo12_service:
            raise HTTPException(
                status_code=503,
                detail="YOLO12 service not available"
            )
        
        # Load new model
        await yolo12_service.load_model(model_size, task)
        
        return {
            "status": "success",
            "message": f"✅ Successfully switched to {model_size.upper()} for {task.upper()} task",
            "current_model": model_size,
            "current_task": task,
            "model_info": yolo12_service.get_model_info()
        }
        
    except Exception as e:
        logger.error(f"Model switch error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to switch model: {str(e)}"
        )

# YOLO12 classification endpoint (unified detection + analysis)
@app.post("/yolo12/classify", tags=["🎯 YOLO12 Detection"])
async def yolo12_classify_image(
    file: UploadFile = File(..., description="Image file for YOLO12 classification"),
    expected_object: str = Form(None, description="Expected object to detect"),
    confidence_threshold: float = Form(default=0.25, description="Confidence threshold")
):
    """
    🔍 **YOLO12 Unified Classification**
    
    Perform object detection with additional analysis for expected objects.
    Perfect for verification workflows and audit automation.
    """
    try:
        if not YOLO12_AVAILABLE or not yolo12_service:
            raise HTTPException(
                status_code=503,
                detail="YOLO12 service not available"
            )
        
        # Read image bytes
        image_bytes = await file.read()
        
        # Perform YOLO12 classification
        result = await yolo12_service.classify_with_yolo12(
            image_bytes=image_bytes,
            expected_object=expected_object,
            confidence_threshold=confidence_threshold
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"YOLO12 classification error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"YOLO12 classification failed: {str(e)}"
        )

# Model information endpoint
@app.get("/yolo12/info", tags=["📋 YOLO12 Information"])
async def get_yolo12_info():
    """
    📋 **YOLO12 Model Information**
    
    Get comprehensive information about the loaded YOLO12 model and capabilities.
    """
    if not YOLO12_AVAILABLE or not yolo12_service:
        return {
            "status": "unavailable",
            "message": "YOLO12 service not available",
            "installation_required": True
        }
    
    return yolo12_service.get_model_info()

# Health check endpoint
@app.get("/health", tags=["🏢 HISYNC General"])
async def health_check():
    """HISYNC AI YOLO12 system health check"""
    return {
        "status": "healthy" if (YOLO12_AVAILABLE and yolo12_service and yolo12_service.is_loaded) else "unhealthy",
        "yolo12_loaded": yolo12_service.is_loaded if yolo12_service else False,
        "yolo12_available": YOLO12_AVAILABLE,
        "version": "2.0.0",
        "features": ["YOLO12 Attention-Centric Detection", "Area Attention", "R-ELAN", "FlashAttention"]
    }

if __name__ == "__main__":
    uvicorn.run(
        "main_yolo12:app", 
        host="0.0.0.0", 
        port=8001,  # Different port to avoid conflicts
        reload=True,
        log_level="info"
    )
