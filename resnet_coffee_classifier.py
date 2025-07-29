"""
HISYNC AI - Google ResNet v2 Coffee Classifier
Specialized coffee classification using Google's state-of-the-art ResNet v2 model

© 2025 Hire Synchronisation Pvt. Ltd.
Developer: Abhishek Rajput (@abhi-hisync)
"""

import logging
import time
import asyncio
from typing import Dict, List, Tuple, Any
import numpy as np
from PIL import Image
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import tensorflow as tf
    import tensorflow_hub as hub
    TENSORFLOW_AVAILABLE = True
    logger.info("✅ TensorFlow Hub and TensorFlow loaded for Google ResNet v2")
except ImportError:
    TENSORFLOW_AVAILABLE = False
    logger.warning("⚠️ TensorFlow not available, using simulation mode")

class GoogleResNetCoffeeClassifier:
    """
    Advanced Coffee Classification using Google's ResNet v2
    Optimized for Bluetokie Coffee Bean Roaster
    """
    
    def __init__(self):
        self.model = None
        self.is_loaded = False
        self.model_url = "https://tfhub.dev/google/imagenet/resnet_v2_152/classification/5"
        self.input_size = (224, 224)
        
        # Coffee-specific mappings from ImageNet classes
        self.coffee_class_mappings = {
            967: "espresso",
            968: "cup", 
            969: "coffee_mug",
            504: "coffee_pot",
            # Add more ImageNet indices that relate to coffee
        }
        
        # Enhanced coffee categories for better classification
        self.coffee_keywords = [
            "coffee", "espresso", "cappuccino", "latte", "americano", "macchiato",
            "beans", "roasted", "grind", "brew", "barista", "cafe", "shop",
            "cup", "mug", "pot", "machine", "grinder", "filter", "milk", "foam",
            "art", "roaster", "plantation", "arabica", "robusta", "blend"
        ]
            logger.info("🔥 Loading HISYNC AI ResNet-V2 Coffee Classification Model...")
            
            # Try to load ResNet-V2 from Kaggle first, fallback to TensorFlow Hub
            try:
                logger.info("Attempting to load ResNet-V2 from Kaggle Models...")
                self.model = hub.load(self.resnet_url)
                logger.info("✅ Successfully loaded ResNet-V2 from Kaggle!")
            except Exception as e:
                logger.warning(f"Kaggle model loading failed: {e}")
                logger.info("Loading ResNet-V2 from TensorFlow Hub (fallback)...")
                self.model = hub.load(self.fallback_url)
                logger.info("✅ Successfully loaded ResNet-V2 from TensorFlow Hub!")
            
            # Load enhanced class labels with coffee focus
            self.class_labels = self._load_imagenet_labels()
            self.coffee_labels = self._load_coffee_cafe_labels()
            
            self.is_loaded = True
            logger.info("🎉 HISYNC AI ResNet-V2 Coffee Model loaded successfully!")
            logger.info(f"📊 Model ready for Bluetokie coffee verification!")
            
        except Exception as e:
            logger.error(f"❌ Failed to load HISYNC AI ResNet-V2 model: {str(e)}")
            # Fallback to MobileNetV2 if ResNet-V2 fails
            logger.info("🔄 Falling back to MobileNetV2...")
            try:
                self.model = tf.keras.applications.MobileNetV2(
                    weights='imagenet',
                    include_top=True,
                    input_shape=(224, 224, 3)
                )
                self.class_labels = self._load_imagenet_labels()
                self.coffee_labels = self._load_coffee_cafe_labels()
                self.is_loaded = True
                logger.info("✅ MobileNetV2 fallback loaded successfully!")
            except Exception as fallback_error:
                raise Exception(f"Both ResNet-V2 and MobileNetV2 loading failed: {str(fallback_error)}")
    
    def _load_coffee_cafe_labels(self) -> Dict[str, List[str]]:
        """Load comprehensive coffee and cafe labels for Bluetokie verification"""
        return {
            # Coffee Equipment & Beans (Bluetokie Priority)
            'coffee_beans': [
                'coffee', 'bean', 'espresso', 'roasted', 'arabica', 'robusta',
                'sack', 'bag', 'grind', 'ground', 'roast', 'cafe'
            ],
            'coffee_machine': [
                'espresso machine', 'coffee machine', 'machine', 'espresso',
                'coffee maker', 'brewing', 'steam', 'barista', 'commercial'
            ],
            'coffee_grinder': [
                'grinder', 'mill', 'coffee grinder', 'burr', 'grinding', 'blade'
            ],
            'coffee_equipment': [
                'french press', 'pour over', 'filter', 'chemex', 'v60',
                'aeropress', 'moka pot', 'brewing', 'dripper', 'kettle'
            ],
            
            # Coffee Beverages
            'espresso_drinks': [
                'espresso', 'shot', 'doppio', 'ristretto', 'crema',
                'espresso cup', 'small cup', 'demitasse'
            ],
            'cappuccino': [
                'cappuccino', 'foam', 'milk foam', 'steamed milk',
                'coffee foam', 'foam art', 'cap', 'milk'
            ],
            'latte': [
                'latte', 'latte art', 'milk coffee', 'flat white',
                'cortado', 'macchiato', 'coffee art', 'heart', 'leaf'
            ],
            'cold_coffee': [
                'iced coffee', 'cold brew', 'nitro coffee', 'iced',
                'cold', 'ice coffee', 'frappuccino', 'frappe'
            ],
            
            # Cafe Environment
            'cafe_space': [
                'coffee shop', 'cafe', 'coffeehouse', 'coffee bar',
                'seating', 'table', 'chair', 'counter', 'restaurant'
            ],
            'coffee_service': [
                'menu', 'board', 'price', 'coffee menu', 'chalkboard',
                'service', 'barista', 'server', 'staff'
            ],
            'coffee_accessories': [
                'coffee cup', 'mug', 'cup', 'saucer', 'glass',
                'takeaway', 'paper cup', 'travel mug', 'sleeve'
            ],
            'pastries_food': [
                'croissant', 'muffin', 'pastry', 'danish', 'cake',
                'cookie', 'biscotti', 'scone', 'donut', 'bagel', 'bread'
            ],
            
            # Professional Equipment (Bluetokie Focus)
            'commercial_equipment': [
                'commercial', 'professional', 'industrial', 'roaster',
                'roasting', 'equipment', 'machinery', 'steel'
            ],
            'packaging_branding': [
                'package', 'bag', 'label', 'brand', 'packaging',
                'product', 'retail', 'coffee bag', 'box'
            ]
        }
    
    def _load_imagenet_labels(self) -> Dict[int, str]:
        """Load ImageNet class labels with enhanced coffee/cafe detection"""
        # Core ImageNet labels with coffee/food focus
        labels = {
            # Beverages
            967: 'espresso', 968: 'coffee', 969: 'tea', 504: 'coffee_mug',
            # Food items  
            947: 'croissant', 948: 'bagel', 949: 'pretzel', 950: 'pizza',
            951: 'hotdog', 952: 'taco', 953: 'burrito', 954: 'plate',
            955: 'guacamole', 956: 'consomme', 957: 'hot_pot',
            958: 'trifle', 959: 'ice_cream', 960: 'ice_lolly',
            961: 'french_loaf', 962: 'bagel', 963: 'pretzel',
            # Objects
            639: 'coffee_mug', 640: 'cup', 641: 'wine_glass', 642: 'bottle',
            643: 'wine_bottle', 644: 'cocktail_shaker', 645: 'can_opener',
            # Kitchen equipment
            567: 'coffee_grinder', 568: 'espresso_machine', 569: 'blender',
            570: 'coffee_pot', 571: 'teapot', 572: 'pressure_cooker',
            # Furniture
            423: 'barber_chair', 424: 'chair', 425: 'dining_table',
            426: 'table', 427: 'coffee_table', 428: 'desk',
        }
        
        # Add more ImageNet labels for broader recognition
        standard_labels = {
            # Common objects that might appear in cafes
            0: 'background', 1: 'person', 2: 'bicycle', 3: 'car',
            4: 'motorcycle', 5: 'airplane', 6: 'bus', 7: 'train',
            8: 'truck', 9: 'boat', 10: 'traffic_light',
            # Add animals (some ImageNet classes)
            281: 'tabby_cat', 282: 'tiger_cat', 283: 'persian_cat',
            151: 'chihuahua', 152: 'japanese_spaniel', 207: 'golden_retriever',
        }
        
        # Merge dictionaries
        all_labels = {**standard_labels, **labels}
        
        # Create simplified labels
        simplified_labels = {}
        for idx, label in all_labels.items():
            simplified_labels[idx] = label.lower().replace('_', ' ')
            
        return simplified_labels
    
    def _preprocess_image_resnet(self, image_bytes: bytes) -> tf.Tensor:
        """Preprocess image for ResNet-V2 model"""
        try:
            # Open and convert image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # ResNet-V2 typically uses 224x224 input
            image = image.resize((224, 224))
            
            # Convert to tensor and normalize for ResNet-V2
            img_array = np.array(image, dtype=np.float32)
            img_array = np.expand_dims(img_array, axis=0)
            
            # ResNet-V2 preprocessing (normalize to [-1, 1])
            img_array = (img_array / 127.5) - 1.0
            
            return tf.convert_to_tensor(img_array)
            
        except Exception as e:
            raise ValueError(f"HISYNC AI ResNet-V2 preprocessing failed: {str(e)}")
    
    def _analyze_coffee_relevance_enhanced(self, predicted_classes: List[str], confidence_scores: List[float]) -> Dict[str, Any]:
        """Enhanced coffee/cafe relevance analysis with confidence weighting"""
        relevance_score = 0.0
        matched_categories = []
        coffee_indicators = []
        confidence_weighted_score = 0.0
        
        # Convert predictions to lowercase for analysis
        pred_text = " ".join(predicted_classes).lower()
        
        # Check against coffee/cafe categories with confidence weighting
        for category, keywords in self.coffee_labels.items():
            category_score = 0
            matched_keywords = []
            weighted_score = 0
            
            for i, keyword in enumerate(keywords):
                for j, pred_class in enumerate(predicted_classes):
                    if keyword.lower() in pred_class.lower():
                        category_score += 1
                        matched_keywords.append(keyword)
                        # Weight by prediction confidence
                        if j < len(confidence_scores):
                            weighted_score += confidence_scores[j]
            
            if category_score > 0:
                relevance_score += category_score * 0.1
                confidence_weighted_score += weighted_score * 0.1
                matched_categories.append({
                    "category": category,
                    "score": category_score,
                    "weighted_score": weighted_score,
                    "keywords": matched_keywords
                })
        
        # Bonus scoring for direct coffee/cafe terms
        coffee_terms = ['coffee', 'cafe', 'espresso', 'latte', 'cappuccino', 'barista', 'roasted', 'beans']
        for term in coffee_terms:
            if term in pred_text:
                relevance_score += 0.3
                coffee_indicators.append(term)
        
        # Enhanced scoring with ResNet-V2 confidence
        if confidence_scores:
            avg_confidence = np.mean(confidence_scores[:3])  # Top 3 predictions
            confidence_weighted_score = (confidence_weighted_score + relevance_score) * avg_confidence
        
        # Normalize scores
        relevance_score = min(1.0, relevance_score)
        confidence_weighted_score = min(1.0, confidence_weighted_score)
        
        return {
            "relevance_score": relevance_score,
            "confidence_weighted_score": confidence_weighted_score,
            "is_coffee_related": relevance_score >= 0.3,
            "matched_categories": matched_categories,
            "coffee_indicators": coffee_indicators,
            "confidence_level": "high" if confidence_weighted_score >= 0.7 else "medium" if confidence_weighted_score >= 0.4 else "low",
            "resnet_enhanced": True
        }
    
    async def classify_coffee_resnet(
        self, 
        image_bytes: bytes, 
        expected_label: str, 
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        HISYNC AI - ResNet-V2 Enhanced Coffee & Cafe Classification
        Superior accuracy with Google's ResNet-V2 model
        """
        start_time = time.time()
        
        try:
            if not self.is_loaded:
                raise Exception("HISYNC AI ResNet-V2 model not loaded. Please initialize the service first.")
            
            # Validate image
            self._validate_image(image_bytes)
            
            # Preprocess for ResNet-V2
            processed_image = self._preprocess_image_resnet(image_bytes)
            
            # Make prediction with ResNet-V2
            if hasattr(self.model, 'signatures'):
                # TensorFlow Hub model
                predictions = self.model.signatures['serving_default'](processed_image)
                if isinstance(predictions, dict):
                    predictions = list(predictions.values())[0]
                predictions = predictions.numpy()
            else:
                # Keras model (fallback)
                predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top 5 predictions
            if len(predictions.shape) > 1:
                predictions = predictions[0]
            
            top_indices = np.argsort(predictions)[-5:][::-1]
            
            # Format predictions with confidence scores
            all_predictions = []
            predicted_classes = []
            confidence_scores = []
            
            for idx in top_indices:
                class_name = self.class_labels.get(int(idx), f"class_{idx}")
                confidence = float(predictions[idx])
                
                all_predictions.append({
                    "label": class_name,
                    "confidence": confidence,
                    "class_id": int(idx)
                })
                predicted_classes.append(class_name)
                confidence_scores.append(confidence)
            
            # Enhanced coffee analysis with ResNet-V2 confidence
            coffee_analysis = self._analyze_coffee_relevance_enhanced(predicted_classes, confidence_scores)
            
            # Enhanced matching for coffee items
            is_match, matched_class, match_confidence = self._find_best_coffee_match(
                predicted_classes, expected_label
            )
            
            # Primary prediction
            primary_prediction = all_predictions[0]
            confidence_met = primary_prediction["confidence"] >= confidence_threshold
            
            # Enhanced status with ResNet-V2 insights
            if is_match and confidence_met and coffee_analysis["is_coffee_related"]:
                status = "correct"
                message = f"✅ HISYNC AI ResNet-V2 SUCCESS! Detected '{primary_prediction['label']}' matches '{expected_label}' with {primary_prediction['confidence']:.2%} confidence. Coffee relevance: {coffee_analysis['confidence_level'].upper()}"
            elif is_match and coffee_analysis["is_coffee_related"]:
                status = "partial"
                message = f"⚠️ HISYNC AI ResNet-V2 PARTIAL MATCH. '{primary_prediction['label']}' matches '{expected_label}' but confidence {primary_prediction['confidence']:.2%} < {confidence_threshold:.2%}"
            elif coffee_analysis["is_coffee_related"]:
                status = "coffee_detected"
                message = f"☕ HISYNC AI ResNet-V2 COFFEE DETECTED. Found coffee context but '{primary_prediction['label']}' != '{expected_label}'. Relevance: {coffee_analysis['confidence_level'].upper()}"
            else:
                status = "incorrect"
                message = f"❌ HISYNC AI ResNet-V2: '{primary_prediction['label']}' doesn't match coffee item '{expected_label}'. Consider image quality or expected label."
            
            processing_time = (time.time() - start_time) * 1000
            
            return {
                "status": status,
                "expected_label": expected_label,
                "prediction_result": {
                    "predicted_label": primary_prediction["label"],
                    "confidence": primary_prediction["confidence"],
                    "all_predictions": all_predictions,
                    "model_type": "ResNet-V2" if "signatures" in dir(self.model) else "MobileNetV2-Fallback"
                },
                "coffee_analysis": coffee_analysis,
                "is_match": is_match,
                "confidence_met": confidence_met,
                "message": message,
                "processing_time_ms": processing_time,
                "bluetokie_verification": {
                    "is_coffee_related": coffee_analysis["is_coffee_related"],
                    "relevance_score": coffee_analysis["confidence_weighted_score"],
                    "matched_categories": coffee_analysis["matched_categories"],
                    "recommendation": self._get_resnet_recommendation(status, coffee_analysis),
                    "model_used": "Google ResNet-V2 (Enhanced)"
                }
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"HISYNC AI ResNet-V2 error: {str(e)}")
            
            return {
                "status": "error",
                "expected_label": expected_label,
                "prediction_result": {
                    "predicted_label": "error",
                    "confidence": 0.0,
                    "all_predictions": [],
                    "model_type": "error"
                },
                "coffee_analysis": {"relevance_score": 0.0, "is_coffee_related": False, "resnet_enhanced": False},
                "is_match": False,
                "confidence_met": False,
                "message": f"❌ HISYNC AI ResNet-V2 Classification failed: {str(e)}",
                "processing_time_ms": processing_time,
                "bluetokie_verification": {
                    "is_coffee_related": False,
                    "recommendation": "Please check image quality and try again. Contact support@hisync.in if issue persists."
                }
            }
    
    def _find_best_coffee_match(self, predicted_classes: List[str], expected_label: str) -> Tuple[bool, str, float]:
        """Enhanced coffee matching with ResNet-V2 intelligence"""
        expected_normalized = expected_label.lower().strip()
        
        # Direct exact match
        for pred_class in predicted_classes:
            if expected_normalized == pred_class.lower().strip():
                return True, pred_class, 1.0
        
        # Coffee-specific intelligent matching
        for category, keywords in self.coffee_labels.items():
            for keyword in keywords:
                if keyword.lower() in expected_normalized:
                    for pred_class in predicted_classes:
                        if keyword.lower() in pred_class.lower():
                            return True, pred_class, 0.9
        
        # Substring matching
        for pred_class in predicted_classes:
            pred_normalized = pred_class.lower()
            if expected_normalized in pred_normalized or pred_normalized in expected_normalized:
                return True, pred_class, 0.8
        
        # Word-based matching with coffee context
        expected_words = expected_normalized.split()
        for word in expected_words:
            if len(word) > 2:
                for pred_class in predicted_classes:
                    if word in pred_class.lower():
                        return True, pred_class, 0.6
        
        return False, predicted_classes[0] if predicted_classes else "unknown", 0.0
    
    def _get_resnet_recommendation(self, status: str, coffee_analysis: Dict) -> str:
        """Generate ResNet-V2 enhanced recommendations"""
        if status == "correct":
            return "✅ RESNET-V2 APPROVED: High-confidence coffee verification. Excellent for Bluetokie standards."
        elif status == "partial":
            return "⚠️ RESNET-V2 REVIEW: Coffee detected but confidence below threshold. Consider retaking image."
        elif status == "coffee_detected":
            return "☕ RESNET-V2 COFFEE CONTEXT: Coffee elements found but item mismatch. Verify expected label."
        else:
            if coffee_analysis.get("relevance_score", 0) > 0:
                return "🔍 RESNET-V2 PARTIAL COFFEE: Some coffee elements detected. Improve lighting or angle."
            else:
                return "❌ RESNET-V2 NON-COFFEE: No coffee context detected. Ensure image contains coffee/cafe items."
    
    def _validate_image(self, image_bytes: bytes) -> None:
        """Validate image for ResNet-V2 processing"""
        if len(image_bytes) > self.max_file_size:
            raise ValueError(f"Image exceeds {self.max_file_size/1024/1024}MB limit")
        
        try:
            image = Image.open(io.BytesIO(image_bytes))
            if image.format not in self.supported_formats:
                raise ValueError(f"Unsupported format: {image.format}")
        except Exception as e:
            raise ValueError(f"Invalid image: {str(e)}")


# Global ResNet-V2 service instance for Bluetokie
resnet_coffee_service = HISYNCResNetCoffeeClassifier()
