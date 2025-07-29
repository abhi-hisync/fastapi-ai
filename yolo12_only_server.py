#!/usr/bin/env python3
"""
🎯 HISYNC AI - YOLO12 Only Server
Pure YOLO12 FastAPI server without TensorFlow conflicts
"""

import os
import logging
import time
from typing import Optional
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import YOLO12
YOLO12_AVAILABLE = False
yolo12_service = None

try:
    from yolo12_classifier import YOLO12ClassificationService
    yolo12_service = YOLO12ClassificationService()
    YOLO12_AVAILABLE = True
    logger.info("✅ YOLO12 packages available!")
except ImportError as e:
    logger.warning(f"⚠️ YOLO12 not available: {str(e)}")
    yolo12_service = None

# Initialize FastAPI app
app = FastAPI(
    title="🎯 HISYNC AI - YOLO12 Only Server",
    description="Pure YOLO12 Attention-Centric Object Detection API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting HISYNC AI YOLO12 Server...")
    
    # Initialize YOLO12
    if YOLO12_AVAILABLE and yolo12_service:
        await yolo12_service.load_model()
        logger.info("✅ HISYNC AI YOLO12 Attention-Centric Object Detector loaded successfully!")
    else:
        logger.warning("⚠️ YOLO12 not available, using simulation mode")
        logger.info("✅ YOLO12 simulation mode ready!")
    
    logger.info("🎯 HISYNC AI YOLO12 Server ready!")

@app.get("/", tags=["🏠 Home"])
async def root():
    """Welcome endpoint"""
    return {
        "message": "🎯 HISYNC AI - YOLO12 Only Server",
        "status": "active",
        "yolo12_available": YOLO12_AVAILABLE,
        "endpoints": {
            "docs": "/docs",
            "detect": "/yolo12/detect",
            "classify": "/yolo12/classify",
            "info": "/yolo12/info"
        }
    }

@app.get("/health", tags=["🏥 Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "yolo12_available": YOLO12_AVAILABLE,
        "yolo12_loaded": yolo12_service.is_loaded if yolo12_service else False
    }

# YOLO12 Detection endpoint
@app.post("/yolo12/detect", tags=["🎯 YOLO12 Detection"])
async def yolo12_detect_objects(
    file: UploadFile = File(..., description="Image file for YOLO12 object detection"),
    confidence_threshold: float = Form(default=0.25, description="Confidence threshold (0.1-1.0)"),
    iou_threshold: float = Form(default=0.45, description="IoU threshold for NMS (0.1-1.0)")
):
    """
    🎯 **YOLO12 Object Detection**
    
    Advanced attention-centric object detection using YOLO12 architecture.
    """
    try:
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="Invalid file type")
        
        if not 0.1 <= confidence_threshold <= 1.0:
            raise HTTPException(status_code=400, detail="Confidence threshold must be between 0.1 and 1.0")
        
        image_bytes = await file.read()
        
        if YOLO12_AVAILABLE and yolo12_service and yolo12_service.is_loaded:
            result = await yolo12_service.detect_objects(
                image_bytes=image_bytes,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
        else:
            # Simulation mode
            result = {
                "status": "simulation",
                "message": "🔄 YOLO12 Detection in simulation mode",
                "detections": [
                    {
                        "class": "cup",
                        "confidence": 0.87,
                        "bbox": {"x1": 120, "y1": 180, "x2": 220, "y2": 280}
                    }
                ],
                "simulation_mode": True
            }
        
        return result
        
    except Exception as e:
        logger.error(f"YOLO12 detection error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

# YOLO12 Classification endpoint
@app.post("/yolo12/classify", tags=["🎯 YOLO12 Classification"])
async def yolo12_classify_image(
    file: UploadFile = File(..., description="Image file for YOLO12 classification"),
    expected_object: str = Form(None, description="Expected object to detect"),
    confidence_threshold: float = Form(default=0.25, description="Confidence threshold (0.1-1.0)")
):
    """
    🔍 **YOLO12 Unified Classification**
    
    Advanced classification with object detection and analysis.
    """
    try:
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
            # Enhanced simulation
            result = {
                "status": "simulation",
                "message": "🔄 YOLO12 Classification in simulation mode",
                "detections": [
                    {
                        "class": "cup",
                        "confidence": 0.87,
                        "is_coffee_related": True
                    }
                ],
                "coffee_analysis": {
                    "is_cafe_environment": True,
                    "coffee_context_score": 0.92
                },
                "simulation_mode": True
            }
        
        return result
        
    except Exception as e:
        logger.error(f"YOLO12 classification error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")

@app.get("/yolo12/info", tags=["🎯 YOLO12 Info"])
async def yolo12_model_info():
    """Get YOLO12 model information"""
    if YOLO12_AVAILABLE and yolo12_service:
        return yolo12_service.get_model_info()
    else:
        return {
            "status": "simulation_mode",
            "model_name": "YOLO12 (Simulation)",
            "available": False,
            "note": "Install ultralytics package for real YOLO12"
        }

if __name__ == "__main__":
    logger.info("🎯 Starting HISYNC AI YOLO12 Only Server...")
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,  # Different port to avoid conflicts
        log_level="info"
    )
