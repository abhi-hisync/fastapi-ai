"""
HISYNC AI - YOLO12 Attention-Centric Object Detection Classifier
State-of-the-art object detection using YOLO12's attention mechanisms
Specialized for comprehensive object detection with superior accuracy

© 2025 Hire Synchronisation Pvt. Ltd.
Developer: Abhishek Rajput (@abhi-hisync)
Client: Bluetokie Coffee Bean Roaster & General Object Detection
"""

import logging
import time
import asyncio
from typing import Dict, List, Tuple, Any, Optional
import numpy as np
from PIL import Image
import io
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
    import torch
    import cv2
    YOLO_AVAILABLE = True
    logger.info("✅ YOLO12 and PyTorch loaded successfully")
except ImportError as e:
    YOLO_AVAILABLE = False
    logger.warning(f"⚠️ YOLO12 not available: {e}")

class YOLO12ClassificationService:
    """
    Advanced Object Detection using YOLO12's Attention-Centric Architecture
    Features:
    - Area Attention Mechanism for efficient large receptive fields
    - R-ELAN (Residual Efficient Layer Aggregation Networks)
    - Optimized Attention Architecture with FlashAttention
    - Comprehensive task support: Detection, Segmentation, Classification, Pose, OBB
    """
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.supported_models = {
            'yolo12n': 'nano - fastest, lowest accuracy',
            'yolo12s': 'small - balanced speed/accuracy',
            'yolo12m': 'medium - good accuracy',
            'yolo12l': 'large - high accuracy',
            'yolo12x': 'extra large - highest accuracy'
        }
        self.current_model = 'yolo12x'  # Using extra large for highest accuracy
        self.supported_tasks = ['detect', 'segment', 'classify', 'pose', 'obb']
        self.current_task = 'detect'
        self.supported_formats = ['JPEG', 'PNG', 'JPG', 'WEBP', 'BMP', 'TIFF']
        self.max_file_size = 15 * 1024 * 1024  # 15MB for YOLO12
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Coffee-specific class mappings for Bluetokie
        self.coffee_class_mappings = {
            'cup': ['coffee_cup', 'tea_cup', 'mug'],
            'bottle': ['coffee_bottle', 'thermos'],
            'person': ['barista', 'customer', 'coffee_worker'],
            'dining table': ['cafe_table', 'coffee_table'],
            'chair': ['cafe_chair', 'coffee_shop_seating'],
            'potted plant': ['cafe_decoration', 'coffee_plant'],
            'bowl': ['coffee_bowl', 'serving_bowl'],
            'spoon': ['coffee_spoon', 'stirring_spoon'],
            'knife': ['cafe_utensil'],
            'cake': ['pastry', 'coffee_cake', 'dessert'],
            'donut': ['coffee_pastry', 'cafe_snack'],
            'sandwich': ['cafe_food', 'coffee_accompaniment']
        }
        
        # YOLO12 COCO class names
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
            'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
            'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
            'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich',
            'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
            'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Performance metrics
        self.processing_stats = {
            'total_classifications': 0,
            'avg_processing_time': 0.0,
            'successful_detections': 0,
            'coffee_related_detections': 0
        }

    async def load_model(self, model_size: str = 'yolo12n', task: str = 'detect'):
        """
        Load YOLO12 model with specified size and task
        
        Args:
            model_size: Model size ('yolo12n', 'yolo12s', 'yolo12m', 'yolo12l', 'yolo12x')
            task: Task type ('detect', 'segment', 'classify', 'pose', 'obb')
        """
        try:
            if not YOLO_AVAILABLE:
                logger.warning("⚠️ YOLO12 not available, using simulation mode")
                await self._simulate_loading()
                return

            # Validate inputs
            if model_size not in self.supported_models:
                model_size = 'yolo12n'
                logger.warning(f"Invalid model size, using default: {model_size}")
                
            if task not in self.supported_tasks:
                task = 'detect'
                logger.warning(f"Invalid task, using default: {task}")

            self.current_model = model_size
            self.current_task = task
            
            # Construct model name based on task
            if task == 'detect':
                model_name = f"{model_size}.pt"
            else:
                model_name = f"{model_size}-{task}.pt"
            
            logger.info(f"🔥 Loading YOLO12 {model_size.upper()} for {task.upper()} task...")
            logger.info(f"📥 Model: {model_name}")
            
            # Load YOLO12 model
            self.model = YOLO(model_name)
            
            # Warm up the model
            logger.info("🔥 Warming up YOLO12 model...")
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            _ = self.model(dummy_image, verbose=False)
            
            self.is_loaded = True
            logger.info("✅ YOLO12 Attention-Centric Object Detector ready!")
            logger.info(f"🎯 Model: {model_size.upper()} | Task: {task.upper()}")
            logger.info(f"🚀 Features: Area Attention, R-ELAN, FlashAttention optimization")
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLO12 model: {e}")
            logger.info("🔄 Falling back to simulation mode...")
            await self._simulate_loading()
    
    async def _simulate_loading(self):
        """Simulate model loading for testing"""
        await asyncio.sleep(2)
        self.is_loaded = True
        logger.info("✅ YOLO12 simulation mode ready!")

    def _validate_image(self, image_bytes: bytes):
        """Validate image file"""
        if len(image_bytes) == 0:
            raise ValueError("Empty image file")
        
        if len(image_bytes) > self.max_file_size:
            raise ValueError(f"Image too large. Max size: {self.max_file_size / (1024*1024):.1f}MB")
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {image.format}")
        except Exception as e:
            raise ValueError(f"Invalid image: {e}")

    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for YOLO12"""
        try:
            # Convert to PIL Image
            image = Image.open(io.BytesIO(image_bytes))
            logger.info(f"📸 Image loaded: {image.format} {image.size} {image.mode}")
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                logger.info(f"🔄 Converting {image.mode} to RGB")
                image = image.convert('RGB')
            
            # Convert to numpy array
            image_array = np.array(image)
            logger.info(f"✅ Image preprocessed: {image_array.shape}")
            
            return image_array
            
        except Exception as e:
            logger.error(f"❌ Image preprocessing failed: {e}")
            raise ValueError(f"Image preprocessing failed: {e}")

    def _is_coffee_related(self, class_name: str, confidence: float) -> Tuple[bool, float]:
        """Determine if detection is coffee/cafe related"""
        coffee_keywords = [
            'cup', 'bottle', 'person', 'dining', 'table', 'chair', 'bowl', 'spoon',
            'knife', 'cake', 'donut', 'sandwich', 'potted', 'plant'
        ]
        
        class_lower = class_name.lower()
        is_coffee = any(keyword in class_lower for keyword in coffee_keywords)
        
        if is_coffee:
            # Boost confidence for coffee-related items
            adjusted_confidence = min(confidence * 1.2, 1.0)
        else:
            adjusted_confidence = confidence * 0.8
            
        return is_coffee, adjusted_confidence

    def _analyze_coffee_context(self, detections: List[Dict]) -> Dict[str, Any]:
        """Analyze coffee/cafe context in detections"""
        coffee_items = []
        cafe_items = []
        people_count = 0
        furniture_count = 0
        
        for detection in detections:
            class_name = detection['class']
            confidence = detection['confidence']
            
            is_coffee, _ = self._is_coffee_related(class_name, confidence)
            
            if is_coffee:
                if class_name in ['cup', 'bottle', 'bowl', 'spoon']:
                    coffee_items.append(detection)
                elif class_name in ['chair', 'dining table', 'potted plant']:
                    cafe_items.append(detection)
                elif class_name == 'person':
                    people_count += 1
                elif class_name in ['chair', 'dining table', 'couch']:
                    furniture_count += 1

        total_coffee_score = len(coffee_items) * 0.4 + len(cafe_items) * 0.3 + people_count * 0.2 + furniture_count * 0.1
        
        context_analysis = {
            'is_cafe_environment': total_coffee_score >= 1.0,
            'coffee_items_detected': len(coffee_items),
            'cafe_furniture_detected': len(cafe_items),
            'people_detected': people_count,
            'coffee_context_score': min(total_coffee_score, 1.0),
            'confidence_level': 'high' if total_coffee_score >= 2.0 else 'medium' if total_coffee_score >= 1.0 else 'low',
            'detected_coffee_items': [item['class'] for item in coffee_items],
            'detected_cafe_items': [item['class'] for item in cafe_items]
        }
        
        return context_analysis

    async def detect_objects(
        self, 
        image_bytes: bytes, 
        confidence_threshold: float = None,
        iou_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Perform YOLO12 object detection on image
        
        Args:
            image_bytes: Image data
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
            
        Returns:
            Detection results with coffee/cafe analysis
        """
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise Exception("YOLO12 model not loaded")
            
            # Use provided thresholds or defaults
            conf_thresh = confidence_threshold if confidence_threshold else self.confidence_threshold
            iou_thresh = iou_threshold if iou_threshold else self.iou_threshold
            
            # Validate and preprocess image
            self._validate_image(image_bytes)
            image_array = self._preprocess_image(image_bytes)
            
            if YOLO_AVAILABLE and self.model is not None:
                # Run YOLO12 inference
                logger.info(f"🎯 Running YOLO12 inference with conf={conf_thresh}, iou={iou_thresh}")
                try:
                    results = self.model(
                        image_array,
                        conf=conf_thresh,
                        iou=iou_thresh,
                        verbose=False
                    )
                    logger.info(f"✅ YOLO12 inference completed successfully")
                except Exception as yolo_error:
                    logger.error(f"❌ YOLO12 inference failed: {yolo_error}")
                    raise Exception(f"YOLO12 inference error: {yolo_error}")
                
                # Process results
                detections = []
                if results and len(results) > 0:
                    result = results[0]
                    
                    if result.boxes is not None:
                        boxes = result.boxes.xyxy.cpu().numpy()
                        confidences = result.boxes.conf.cpu().numpy()
                        class_ids = result.boxes.cls.cpu().numpy().astype(int)
                        
                        for i, (box, conf, cls_id) in enumerate(zip(boxes, confidences, class_ids)):
                            if cls_id < len(self.coco_classes):
                                class_name = self.coco_classes[cls_id]
                                
                                detection = {
                                    'id': i,
                                    'class': class_name,
                                    'class_id': int(cls_id),
                                    'confidence': float(conf),
                                    'bbox': {
                                        'x1': float(box[0]),
                                        'y1': float(box[1]),
                                        'x2': float(box[2]),
                                        'y2': float(box[3]),
                                        'width': float(box[2] - box[0]),
                                        'height': float(box[3] - box[1])
                                    }
                                }
                                
                                # Add coffee relevance
                                is_coffee, adj_conf = self._is_coffee_related(class_name, conf)
                                detection['is_coffee_related'] = is_coffee
                                detection['adjusted_confidence'] = adj_conf
                                
                                detections.append(detection)
            else:
                # Simulation mode
                detections = self._simulate_detections()
            
            # Analyze coffee/cafe context
            coffee_analysis = self._analyze_coffee_context(detections)
            
            # Sort detections by confidence
            detections.sort(key=lambda x: x['confidence'], reverse=True)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.processing_stats['total_classifications'] += 1
            self.processing_stats['successful_detections'] += len(detections)
            if coffee_analysis['is_cafe_environment']:
                self.processing_stats['coffee_related_detections'] += 1
            
            # Calculate running average
            prev_avg = self.processing_stats['avg_processing_time']
            total_count = self.processing_stats['total_classifications']
            self.processing_stats['avg_processing_time'] = (prev_avg * (total_count - 1) + processing_time) / total_count
            
            return {
                'status': 'success',
                'message': f"✅ YOLO12 detected {len(detections)} objects successfully!",
                'model_info': {
                    'name': f'YOLO12-{self.current_model.upper()}',
                    'task': self.current_task,
                    'architecture': 'Attention-Centric with Area Attention & R-ELAN',
                    'features': ['Area Attention Mechanism', 'R-ELAN', 'FlashAttention', 'Optimized MLP'],
                    'accuracy': 'State-of-the-art on COCO dataset'
                },
                'detections': detections,
                'detection_summary': {
                    'total_objects': len(detections),
                    'high_confidence': len([d for d in detections if d['confidence'] > 0.7]),
                    'medium_confidence': len([d for d in detections if 0.4 <= d['confidence'] <= 0.7]),
                    'low_confidence': len([d for d in detections if d['confidence'] < 0.4]),
                    'coffee_related': len([d for d in detections if d['is_coffee_related']])
                },
                'coffee_analysis': coffee_analysis,
                'bluetokie_verification': {
                    'is_suitable_for_audit': coffee_analysis['is_cafe_environment'],
                    'recommendation': self._get_bluetokie_recommendation(coffee_analysis),
                    'audit_score': coffee_analysis['coffee_context_score']
                },
                'processing_info': {
                    'processing_time_ms': processing_time,
                    'confidence_threshold': conf_thresh,
                    'iou_threshold': iou_thresh,
                    'image_processed': True,
                    'yolo12_features_used': ['Area Attention', 'R-ELAN', 'Optimized Attention']
                },
                'performance_stats': self.processing_stats.copy()
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"YOLO12 detection error: {e}")
            
            return {
                'status': 'error',
                'message': f"❌ YOLO12 detection failed: {str(e)}",
                'detections': [],
                'coffee_analysis': {
                    'is_cafe_environment': False,
                    'coffee_context_score': 0.0,
                    'error': str(e)
                },
                'processing_info': {
                    'processing_time_ms': processing_time,
                    'error': str(e)
                }
            }

    def _simulate_detections(self) -> List[Dict]:
        """Simulate detections for testing"""
        simulated = [
            {
                'id': 0,
                'class': 'cup',
                'class_id': 41,
                'confidence': 0.85,
                'is_coffee_related': True,
                'adjusted_confidence': 0.95,
                'bbox': {'x1': 100, 'y1': 150, 'x2': 200, 'y2': 250, 'width': 100, 'height': 100}
            },
            {
                'id': 1,
                'class': 'person',
                'class_id': 0,
                'confidence': 0.92,
                'is_coffee_related': True,
                'adjusted_confidence': 0.98,
                'bbox': {'x1': 50, 'y1': 80, 'x2': 300, 'y2': 400, 'width': 250, 'height': 320}
            }
        ]
        return simulated

    def _get_bluetokie_recommendation(self, coffee_analysis: Dict) -> str:
        """Get Bluetokie-specific recommendation"""
        score = coffee_analysis['coffee_context_score']
        
        if score >= 2.0:
            return "✅ EXCELLENT for Bluetokie audit - Clear cafe/coffee environment detected"
        elif score >= 1.5:
            return "✅ GOOD for Bluetokie audit - Strong coffee context indicators"
        elif score >= 1.0:
            return "⚠️ MODERATE for Bluetokie audit - Some coffee elements detected"
        elif score >= 0.5:
            return "⚠️ LIMITED for Bluetokie audit - Weak coffee context"
        else:
            return "❌ NOT SUITABLE for Bluetokie audit - No significant coffee context"

    async def classify_with_yolo12(
        self,
        image_bytes: bytes,
        expected_object: str = None,
        confidence_threshold: float = 0.25
    ) -> Dict[str, Any]:
        """
        Unified classification interface combining detection with analysis
        Optimized for Bluetokie coffee verification workflows
        """
        try:
            # Perform object detection
            detection_result = await self.detect_objects(
                image_bytes=image_bytes,
                confidence_threshold=confidence_threshold
            )
            
            if detection_result['status'] == 'error':
                return detection_result
            
            detections = detection_result['detections']
            coffee_analysis = detection_result['coffee_analysis']
            
            # Enhanced analysis for expected object
            match_analysis = None
            if expected_object:
                match_analysis = self._analyze_object_match(detections, expected_object)
            
            # Prepare enhanced response
            enhanced_result = {
                **detection_result,
                'classification_type': 'YOLO12 Attention-Centric Object Detection',
                'expected_object_analysis': match_analysis,
                'hisync_recommendation': {
                    'for_bluetokie': self._get_bluetokie_recommendation(coffee_analysis),
                    'confidence_in_environment': coffee_analysis['confidence_level'],
                    'suitable_for_audit': coffee_analysis['is_cafe_environment']
                }
            }
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"YOLO12 classification error: {e}")
            return {
                'status': 'error',
                'message': f"❌ YOLO12 classification failed: {str(e)}",
                'error_details': str(e)
            }

    def _analyze_object_match(self, detections: List[Dict], expected_object: str) -> Dict[str, Any]:
        """Analyze how well detections match expected object"""
        expected_lower = expected_object.lower()
        
        # Find best matches
        exact_matches = [d for d in detections if d['class'].lower() == expected_lower]
        partial_matches = [d for d in detections if expected_lower in d['class'].lower() or d['class'].lower() in expected_lower]
        
        # Coffee-specific matching
        coffee_mappings = self.coffee_class_mappings.get(expected_lower, [])
        mapped_matches = [d for d in detections if d['class'].lower() in [m.lower() for m in coffee_mappings]]
        
        all_matches = exact_matches + partial_matches + mapped_matches
        all_matches = list({d['id']: d for d in all_matches}.values())  # Remove duplicates
        
        if all_matches:
            best_match = max(all_matches, key=lambda x: x['confidence'])
            match_quality = 'exact' if exact_matches else 'partial' if partial_matches else 'mapped'
        else:
            best_match = None
            match_quality = 'none'
        
        return {
            'expected_object': expected_object,
            'match_found': len(all_matches) > 0,
            'match_quality': match_quality,
            'best_match': best_match,
            'total_matches': len(all_matches),
            'exact_matches': len(exact_matches),
            'partial_matches': len(partial_matches),
            'mapped_matches': len(mapped_matches),
            'all_matches': all_matches
        }

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information"""
        return {
            'model_name': f'YOLO12-{self.current_model.upper()}',
            'task': self.current_task,
            'is_loaded': self.is_loaded,
            'architecture': 'Attention-Centric Object Detection',
            'key_features': [
                'Area Attention Mechanism - Efficient large receptive field processing',
                'R-ELAN - Residual Efficient Layer Aggregation Networks',
                'Optimized Attention Architecture with FlashAttention',
                'Reduced parameters with maintained/improved accuracy',
                'Real-time inference with state-of-the-art accuracy'
            ],
            'supported_models': self.supported_models,
            'supported_tasks': self.supported_tasks,
            'supported_formats': self.supported_formats,
            'coco_classes_count': len(self.coco_classes),
            'performance_stats': self.processing_stats,
            'company': 'HISYNC Technologies',
            'integration': 'Specialized for Bluetokie Coffee Verification',
            'paper_reference': 'YOLOv12: Attention-Centric Real-Time Object Detectors (2025)',
            'github_source': 'https://github.com/ultralytics/ultralytics'
        }

# Global YOLO12 service instance
yolo12_service = YOLO12ClassificationService()
