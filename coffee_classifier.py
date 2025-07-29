"""
HISYNC AI - Enhanced Coffee & Cafe Image Classifier
Specialized for Bluetokie Coffee Bean Roaster and Restaurant/Cafe Auditing

© 2025 Hire Synchronisation Pvt. Ltd. All rights reserved.
Developed by: Abhishek Rajput (@abhi-hisync)
Client: Bluetokie - Coffee Bean Roaster Market Leader
"""

import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import time
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json
import re

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HISYNCCoffeeClassificationService:
    """
    HISYNC AI - Enhanced Coffee & Cafe Classification Service
    Specialized for Bluetokie physical verification needs
    """
    
    def __init__(self):
        self.model = None
        self.class_labels = None
        self.coffee_labels = None
        self.is_loaded = False
        self.supported_formats = ['JPEG', 'PNG', 'JPG', 'WEBP']
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.company = "HISYNC Technologies"
        self.product_name = "HISYNC AI - Coffee & Cafe Classification Engine"
        
    async def load_model(self):
        """Load the HISYNC AI optimized model with coffee/cafe focus"""
        try:
            logger.info("Loading HISYNC AI Coffee & Cafe MobileNetV2 model...")
            
            # Load pre-trained MobileNetV2 model (HISYNC optimized)
            self.model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )
            
            # Load enhanced class labels with coffee/cafe focus
            self.class_labels = self._load_imagenet_labels()
            self.coffee_labels = self._load_coffee_cafe_labels()
            
            self.is_loaded = True
            logger.info("HISYNC AI Coffee & Cafe model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load HISYNC AI model: {str(e)}")
            raise Exception(f"HISYNC AI model loading failed: {str(e)}")
    
    def _load_coffee_cafe_labels(self) -> Dict[str, List[str]]:
        """Load comprehensive coffee and cafe labels for Bluetokie verification"""
        return {
            # Coffee Equipment & Beans (Bluetokie Priority)
            'coffee_beans': [
                'coffee', 'bean', 'espresso', 'roasted', 'arabica', 'robusta',
                'sack', 'bag', 'grind', 'ground'
            ],
            'coffee_machine': [
                'espresso machine', 'coffee machine', 'machine', 'espresso',
                'coffee maker', 'brewing', 'steam', 'barista'
            ],
            'coffee_grinder': [
                'grinder', 'mill', 'coffee grinder', 'burr', 'grinding'
            ],
            'coffee_equipment': [
                'french press', 'pour over', 'filter', 'chemex', 'v60',
                'aeropress', 'moka pot', 'brewing', 'dripper'
            ],
            
            # Coffee Beverages
            'espresso_drinks': [
                'espresso', 'shot', 'doppio', 'ristretto', 'crema',
                'espresso cup', 'small cup'
            ],
            'cappuccino': [
                'cappuccino', 'foam', 'milk foam', 'steamed milk',
                'coffee foam', 'foam art'
            ],
            'latte': [
                'latte', 'latte art', 'milk coffee', 'flat white',
                'cortado', 'macchiato', 'coffee art'
            ],
            'cold_coffee': [
                'iced coffee', 'cold brew', 'nitro coffee', 'iced',
                'cold', 'ice coffee', 'frappuccino'
            ],
            
            # Cafe Environment
            'cafe_space': [
                'coffee shop', 'cafe', 'coffeehouse', 'coffee bar',
                'seating', 'table', 'chair', 'counter'
            ],
            'coffee_service': [
                'menu', 'board', 'price', 'coffee menu', 'chalkboard',
                'service', 'barista', 'server'
            ],
            'coffee_accessories': [
                'coffee cup', 'mug', 'cup', 'saucer', 'glass',
                'takeaway', 'paper cup', 'travel mug'
            ],
            'pastries_food': [
                'croissant', 'muffin', 'pastry', 'danish', 'cake',
                'cookie', 'biscotti', 'scone', 'donut', 'bagel'
            ],
            
            # Professional Equipment (Bluetokie Focus)
            'commercial_equipment': [
                'commercial', 'professional', 'industrial', 'roaster',
                'roasting', 'equipment', 'machinery'
            ],
            'packaging_branding': [
                'package', 'bag', 'label', 'brand', 'packaging',
                'product', 'retail', 'coffee bag'
            ]
        }
    
    def _load_imagenet_labels(self) -> Dict[int, str]:
        """Load ImageNet class labels with enhanced coffee/cafe detection"""
        # Enhanced mapping focusing on coffee, food, and restaurant items
        labels = {
            # Food and beverages
            967: 'espresso', 968: 'coffee', 969: 'tea', 970: 'cup',
            # Objects commonly found in cafes
            639: 'coffee_mug', 640: 'cup', 641: 'glass', 642: 'bottle',
            # Kitchen and cafe equipment
            567: 'coffee_grinder', 568: 'espresso_machine', 569: 'blender',
            570: 'coffee_pot', 571: 'teapot', 572: 'pressure_cooker',
            # Furniture and cafe environment
            423: 'barber_chair', 424: 'cafe_chair', 425: 'dining_table',
            426: 'restaurant_table', 427: 'coffee_table',
            # Food items
            947: 'croissant', 948: 'bagel', 949: 'pretzel', 950: 'pizza',
            951: 'hotdog', 952: 'taco', 953: 'burrito', 954: 'plate',
            955: 'guacamole', 956: 'consomme', 957: 'hot_pot',
            958: 'trifle', 959: 'ice_cream', 960: 'ice_lolly',
            961: 'french_loaf', 962: 'bagel', 963: 'pretzel',
        }
        
        # Add standard ImageNet labels for broader recognition
        standard_labels = {
            0: 'tench', 1: 'goldfish', 2: 'great_white_shark', 3: 'tiger_shark',
            4: 'hammerhead', 5: 'electric_ray', 6: 'stingray', 7: 'cock', 8: 'hen',
            9: 'ostrich', 10: 'brambling', 11: 'goldfinch', 12: 'house_finch',
            281: 'tabby_cat', 282: 'tiger_cat', 283: 'persian_cat',
            151: 'chihuahua', 152: 'japanese_spaniel', 207: 'golden_retriever',
            208: 'labrador_retriever', 209: 'chesapeake_bay_retriever',
        }
        
        # Merge dictionaries
        all_labels = {**standard_labels, **labels}
        
        # Create simplified labels
        simplified_labels = {}
        for idx, label in all_labels.items():
            simplified_labels[idx] = label.lower().replace('_', ' ')
            
        return simplified_labels
    
    def _analyze_coffee_cafe_relevance(self, predicted_classes: List[str], image_description: str = "") -> Dict[str, Any]:
        """Analyze how relevant the predictions are to coffee/cafe context"""
        relevance_score = 0.0
        matched_categories = []
        coffee_indicators = []
        
        # Convert predictions to lowercase for analysis
        pred_text = " ".join(predicted_classes).lower()
        combined_text = f"{pred_text} {image_description}".lower()
        
        # Check against coffee/cafe categories
        for category, keywords in self.coffee_labels.items():
            category_score = 0
            matched_keywords = []
            
            for keyword in keywords:
                if keyword.lower() in combined_text:
                    category_score += 1
                    matched_keywords.append(keyword)
            
            if category_score > 0:
                relevance_score += category_score * 0.1
                matched_categories.append({
                    "category": category,
                    "score": category_score,
                    "keywords": matched_keywords
                })
        
        # Bonus scoring for direct coffee/cafe terms
        coffee_terms = ['coffee', 'cafe', 'espresso', 'latte', 'cappuccino', 'barista', 'roasted']
        for term in coffee_terms:
            if term in combined_text:
                relevance_score += 0.2
                coffee_indicators.append(term)
        
        # Normalize score to 0-1 range
        relevance_score = min(1.0, relevance_score)
        
        return {
            "relevance_score": relevance_score,
            "is_coffee_related": relevance_score >= 0.3,
            "matched_categories": matched_categories,
            "coffee_indicators": coffee_indicators,
            "confidence_level": "high" if relevance_score >= 0.7 else "medium" if relevance_score >= 0.3 else "low"
        }
    
    def _find_best_coffee_match(self, predicted_classes: List[str], expected_label: str) -> Tuple[bool, str, float]:
        """Enhanced matching for coffee/cafe items with Bluetokie intelligence"""
        expected_normalized = expected_label.lower().strip()
        
        # Direct match (highest priority)
        for pred_class in predicted_classes:
            pred_normalized = pred_class.lower().strip()
            if expected_normalized == pred_normalized:
                return True, pred_class, 1.0
        
        # Coffee/cafe specific matching
        for category, keywords in self.coffee_labels.items():
            for keyword in keywords:
                if keyword.lower() in expected_normalized:
                    for pred_class in predicted_classes:
                        if keyword.lower() in pred_class.lower():
                            return True, pred_class, 0.9
        
        # Partial substring matching
        for pred_class in predicted_classes:
            pred_normalized = pred_class.lower()
            if expected_normalized in pred_normalized or pred_normalized in expected_normalized:
                return True, pred_class, 0.8
        
        # Word-based matching
        expected_words = expected_normalized.split()
        for word in expected_words:
            if len(word) > 2:  # Skip short words
                for pred_class in predicted_classes:
                    if word in pred_class.lower():
                        return True, pred_class, 0.6
        
        # Coffee context bonus
        coffee_context_words = ['coffee', 'cafe', 'espresso', 'latte', 'cappuccino']
        if any(word in expected_normalized for word in coffee_context_words):
            for pred_class in predicted_classes:
                pred_lower = pred_class.lower()
                if any(word in pred_lower for word in coffee_context_words):
                    return True, pred_class, 0.7
        
        return False, predicted_classes[0] if predicted_classes else "unknown", 0.0
    
    async def classify_coffee_image(
        self, 
        image_bytes: bytes, 
        expected_label: str, 
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        HISYNC AI - Enhanced Coffee & Cafe Image Classification
        Specialized for Bluetokie physical verification
        """
        start_time = time.time()
        
        try:
            # Validate HISYNC AI model is loaded
            if not self.is_loaded:
                raise Exception("HISYNC AI Coffee model not loaded. Please initialize the service first.")
            
            # Validate image
            self._validate_image(image_bytes)
            
            # Preprocess image
            processed_image = self._preprocess_image(image_bytes)
            
            # Make prediction
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top 5 predictions
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            
            # Format predictions
            all_predictions = []
            predicted_classes = []
            
            for idx in top_indices:
                class_name = self.class_labels.get(idx, f"class_{idx}")
                confidence = float(predictions[0][idx])
                
                all_predictions.append({
                    "label": class_name,
                    "confidence": confidence
                })
                predicted_classes.append(class_name)
            
            # Analyze coffee/cafe relevance
            coffee_analysis = self._analyze_coffee_cafe_relevance(predicted_classes)
            
            # Enhanced matching for coffee/cafe items
            is_match, matched_class, match_confidence = self._find_best_coffee_match(
                predicted_classes, expected_label
            )
            
            # Get primary prediction
            primary_prediction = all_predictions[0]
            
            # Determine confidence and status
            confidence_met = primary_prediction["confidence"] >= confidence_threshold
            
            # Enhanced status determination for coffee/cafe context
            if is_match and confidence_met and coffee_analysis["is_coffee_related"]:
                status = "correct"
                message = f"✅ HISYNC AI COFFEE VERIFICATION SUCCESS! Detected '{primary_prediction['label']}' matches expected '{expected_label}' with {primary_prediction['confidence']:.2%} confidence. Coffee relevance: {coffee_analysis['confidence_level']}"
            elif is_match and coffee_analysis["is_coffee_related"]:
                status = "partial"
                message = f"⚠️ HISYNC AI COFFEE MATCH with LOW CONFIDENCE. Detected '{primary_prediction['label']}' matches '{expected_label}' but confidence {primary_prediction['confidence']:.2%} is below threshold {confidence_threshold:.2%}"
            elif coffee_analysis["is_coffee_related"]:
                status = "coffee_detected"
                message = f"☕ HISYNC AI COFFEE CONTEXT DETECTED but doesn't match expected '{expected_label}'. Detected: '{primary_prediction['label']}' with coffee relevance: {coffee_analysis['confidence_level']}"
            else:
                status = "incorrect"
                message = f"❌ HISYNC AI: Detected '{primary_prediction['label']}' doesn't match expected coffee/cafe item '{expected_label}'. Consider rechecking the image or expected label."
            
            # Calculate processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Return enhanced result for Bluetokie verification
            return {
                "status": status,
                "expected_label": expected_label,
                "prediction_result": {
                    "predicted_label": primary_prediction["label"],
                    "confidence": primary_prediction["confidence"],
                    "all_predictions": all_predictions
                },
                "coffee_analysis": coffee_analysis,
                "is_match": is_match,
                "confidence_met": confidence_met,
                "message": message,
                "processing_time_ms": processing_time,
                "bluetokie_verification": {
                    "is_coffee_related": coffee_analysis["is_coffee_related"],
                    "relevance_score": coffee_analysis["relevance_score"],
                    "matched_categories": coffee_analysis["matched_categories"],
                    "recommendation": self._get_bluetokie_recommendation(status, coffee_analysis)
                }
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"HISYNC AI Coffee Classification error: {str(e)}")
            
            return {
                "status": "error",
                "expected_label": expected_label,
                "prediction_result": {
                    "predicted_label": "error",
                    "confidence": 0.0,
                    "all_predictions": []
                },
                "coffee_analysis": {"relevance_score": 0.0, "is_coffee_related": False},
                "is_match": False,
                "confidence_met": False,
                "message": f"❌ HISYNC AI Coffee Classification failed: {str(e)}",
                "processing_time_ms": processing_time,
                "bluetokie_verification": {
                    "is_coffee_related": False,
                    "relevance_score": 0.0,
                    "recommendation": "Please check image quality and try again"
                }
            }
    
    def _get_bluetokie_recommendation(self, status: str, coffee_analysis: Dict) -> str:
        """Generate Bluetokie-specific recommendations"""
        if status == "correct":
            return "✅ APPROVED for Bluetokie verification. Image matches expected coffee/cafe item."
        elif status == "partial":
            return "⚠️ REVIEW REQUIRED. Item detected but confidence below threshold. Manual verification recommended."
        elif status == "coffee_detected":
            return "☕ COFFEE CONTEXT CONFIRMED but item mismatch. Verify expected label or retake image."
        else:
            if coffee_analysis.get("relevance_score", 0) > 0:
                return "🔍 SOME COFFEE ELEMENTS detected. Consider different angle or lighting for better recognition."
            else:
                return "❌ NOT COFFEE/CAFE RELATED. Please ensure image contains coffee or cafe items for Bluetokie verification."
    
    def _validate_image(self, image_bytes: bytes) -> None:
        """Validate image format and size using HISYNC AI standards"""
        if len(image_bytes) > self.max_file_size:
            raise ValueError(f"Image size exceeds HISYNC AI maximum limit of {self.max_file_size/1024/1024}MB")
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.format not in self.supported_formats:
                raise ValueError(f"Unsupported image format: {image.format}. HISYNC AI supports: {', '.join(self.supported_formats)}")
        except Exception as e:
            raise ValueError(f"Invalid image file for HISYNC AI processing: {str(e)}")
    
    def _preprocess_image(self, image_bytes: bytes) -> np.ndarray:
        """Preprocess image for HISYNC AI model prediction"""
        try:
            # Open and convert image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to model input size
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"HISYNC AI image preprocessing failed: {str(e)}")


# Global HISYNC AI Coffee service instance for Bluetokie
coffee_classification_service = HISYNCCoffeeClassificationService()
