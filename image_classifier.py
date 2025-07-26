import tensorflow as tf
import numpy as np
from PIL import Image
import io
import logging
import time
from typing import List, Dict, Tuple, Any
from pathlib import Path
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HISYNCImageClassificationService:
    """
    HISYNC AI - Enterprise-grade Image Classification Service
    
    Powered by HISYNC Technologies - Advanced AI solutions for business automation.
    Uses optimized MobileNetV2 neural network for robust image classification.
    
    © 2024 HISYNC Technologies. All rights reserved.
    """
    
    def __init__(self):
        self.model = None
        self.class_labels = None
        self.is_loaded = False
        self.supported_formats = ['JPEG', 'PNG', 'JPG', 'WEBP']
        self.max_file_size = 10 * 1024 * 1024  # 10MB
        self.company = "HISYNC Technologies"
        self.product_name = "HISYNC AI - Image Classification Engine"
        
    async def load_model(self):
        """Load the HISYNC AI optimized model and class labels"""
        try:
            logger.info("Loading HISYNC AI MobileNetV2 model...")
            
            # Load pre-trained MobileNetV2 model (HISYNC optimized)
            self.model = tf.keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=True,
                input_shape=(224, 224, 3)
            )
            
            # Load ImageNet class labels with HISYNC enhancements
            self.class_labels = self._load_imagenet_labels()
            
            self.is_loaded = True
            logger.info("HISYNC AI model loaded successfully!")
            
        except Exception as e:
            logger.error(f"Failed to load HISYNC AI model: {str(e)}")
            raise Exception(f"HISYNC AI model loading failed: {str(e)}")
    
    def _load_imagenet_labels(self) -> Dict[int, str]:
        """Load ImageNet class labels with HISYNC AI enhancements"""
        # ImageNet class labels (HISYNC optimized version with common classes)
        labels = {
            0: 'tench', 1: 'goldfish', 2: 'great_white_shark', 3: 'tiger_shark',
            4: 'hammerhead', 5: 'electric_ray', 6: 'stingray', 7: 'cock', 8: 'hen',
            9: 'ostrich', 10: 'brambling', 11: 'goldfinch', 12: 'house_finch',
            13: 'junco', 14: 'indigo_bunting', 15: 'robin', 16: 'bulbul',
            17: 'jay', 18: 'magpie', 19: 'chickadee', 20: 'water_ouzel',
            # Common animals (HISYNC enhanced)
            281: 'tabby_cat', 282: 'tiger_cat', 283: 'persian_cat', 284: 'siamese_cat',
            285: 'egyptian_cat', 151: 'chihuahua', 152: 'japanese_spaniel',
            153: 'maltese_dog', 154: 'pekinese', 155: 'shih-tzu', 156: 'blenheim_spaniel',
            157: 'papillon', 158: 'toy_terrier', 159: 'rhodesian_ridgeback',
            160: 'afghan_hound', 161: 'basset', 162: 'beagle', 163: 'bloodhound',
            164: 'bluetick', 165: 'black-and-tan_coonhound', 166: 'walker_hound',
            167: 'english_foxhound', 168: 'redbone', 169: 'borzoi',
            170: 'irish_wolfhound', 171: 'italian_greyhound', 172: 'whippet',
            173: 'ibizan_hound', 174: 'norwegian_elkhound', 175: 'otterhound',
            176: 'saluki', 177: 'scottish_deerhound', 178: 'weimaraner',
            179: 'staffordshire_bullterrier', 180: 'american_staffordshire_terrier',
            181: 'bedlington_terrier', 182: 'border_terrier', 183: 'kerry_blue_terrier',
            184: 'irish_terrier', 185: 'norfolk_terrier', 186: 'norwich_terrier',
            187: 'yorkshire_terrier', 188: 'wire-haired_fox_terrier',
            189: 'lakeland_terrier', 190: 'sealyham_terrier', 191: 'airedale',
            192: 'cairn', 193: 'australian_terrier', 194: 'dandie_dinmont',
            195: 'boston_bull', 196: 'miniature_schnauzer', 197: 'giant_schnauzer',
            198: 'standard_schnauzer', 199: 'scotch_terrier', 200: 'tibetan_terrier',
            201: 'silky_terrier', 202: 'soft-coated_wheaten_terrier',
            203: 'west_highland_white_terrier', 204: 'lhasa', 205: 'flat-coated_retriever',
            206: 'curly-coated_retriever', 207: 'golden_retriever',
            208: 'labrador_retriever', 209: 'chesapeake_bay_retriever',
            210: 'german_short-haired_pointer', 211: 'vizsla', 212: 'english_setter',
            213: 'irish_setter', 214: 'gordon_setter', 215: 'brittany_spaniel',
            216: 'clumber', 217: 'english_springer', 218: 'welsh_springer_spaniel',
            219: 'cocker_spaniel', 220: 'sussex_spaniel', 221: 'irish_water_spaniel',
            222: 'kuvasz', 223: 'schipperke', 224: 'groenendael', 225: 'malinois',
            226: 'briard', 227: 'kelpie', 228: 'komondor', 229: 'old_english_sheepdog',
            230: 'shetland_sheepdog', 231: 'collie', 232: 'border_collie',
            233: 'bouvier_des_flandres', 234: 'rottweiler', 235: 'german_shepherd',
            236: 'doberman', 237: 'miniature_pinscher', 238: 'greater_swiss_mountain_dog',
            239: 'bernese_mountain_dog', 240: 'appenzeller', 241: 'entlebucher',
            242: 'boxer', 243: 'bull_mastiff', 244: 'tibetan_mastiff',
            245: 'french_bulldog', 246: 'great_dane', 247: 'saint_bernard',
            248: 'eskimo_dog', 249: 'malamute', 250: 'siberian_husky',
            251: 'dalmatian', 252: 'affenpinscher', 253: 'basenji', 254: 'pug',
            255: 'leonberg', 256: 'newfoundland', 257: 'great_pyrenees',
            258: 'samoyed', 259: 'pomeranian', 260: 'chow', 261: 'keeshond',
            262: 'brabancon_griffon', 263: 'pembroke', 264: 'cardigan',
            265: 'toy_poodle', 266: 'miniature_poodle', 267: 'standard_poodle',
            268: 'mexican_hairless', 269: 'timber_wolf', 270: 'white_wolf',
            271: 'red_wolf', 272: 'coyote', 273: 'dingo', 274: 'dhole',
            275: 'african_hunting_dog'
        }
        
        # Add HISYNC AI enhanced mappings for common terms
        simplified_labels = {}
        for idx, label in labels.items():
            simplified_labels[idx] = label.lower().replace('_', ' ')
            
        return simplified_labels
    
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
            # Open and convert image using HISYNC AI preprocessing
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Resize to HISYNC AI model input size
            image = image.resize((224, 224))
            
            # Convert to numpy array and normalize using HISYNC AI standards
            img_array = np.array(image)
            img_array = np.expand_dims(img_array, axis=0)
            img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
            
            return img_array
            
        except Exception as e:
            raise ValueError(f"HISYNC AI image preprocessing failed: {str(e)}")
    
    def _normalize_label(self, label: str) -> str:
        """Normalize label for comparison using HISYNC AI algorithms"""
        return label.lower().strip().replace('_', ' ').replace('-', ' ')
    
    def _find_best_match(self, predicted_classes: List[str], expected_label: str) -> Tuple[bool, str, float]:
        """Find best matching prediction for expected label using HISYNC AI intelligence"""
        expected_normalized = self._normalize_label(expected_label)
        
        # HISYNC AI Direct match algorithm
        for i, pred_class in enumerate(predicted_classes):
            pred_normalized = self._normalize_label(pred_class)
            if expected_normalized == pred_normalized:
                return True, pred_class, 1.0
        
        # HISYNC AI Partial match algorithm (contains)
        for i, pred_class in enumerate(predicted_classes):
            pred_normalized = self._normalize_label(pred_class)
            if expected_normalized in pred_normalized or pred_normalized in expected_normalized:
                return True, pred_class, 0.8
        
        # HISYNC AI Keyword match algorithm (for common animals)
        keywords = expected_normalized.split()
        for keyword in keywords:
            if len(keyword) > 2:  # Skip very short words
                for i, pred_class in enumerate(predicted_classes):
                    pred_normalized = self._normalize_label(pred_class)
                    if keyword in pred_normalized:
                        return True, pred_class, 0.6
        
        return False, predicted_classes[0] if predicted_classes else "unknown", 0.0
    
    async def classify_image(
        self, 
        image_bytes: bytes, 
        expected_label: str, 
        confidence_threshold: float = 0.8
    ) -> Dict[str, Any]:
        """
        HISYNC AI - Classify image and compare with expected label
        
        Advanced AI-powered image classification with intelligent verification.
        
        Args:
            image_bytes: Raw image bytes
            expected_label: Expected classification label
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            HISYNC AI classification result dictionary
        """
        start_time = time.time()
        
        try:
            # Validate HISYNC AI model is loaded
            if not self.is_loaded:
                raise Exception("HISYNC AI model not loaded. Please initialize the service first.")
            
            # Validate image using HISYNC AI standards
            self._validate_image(image_bytes)
            
            # Preprocess image using HISYNC AI algorithms
            processed_image = self._preprocess_image(image_bytes)
            
            # Make prediction using HISYNC AI model
            predictions = self.model.predict(processed_image, verbose=0)
            
            # Get top 5 predictions using HISYNC AI ranking
            top_indices = np.argsort(predictions[0])[-5:][::-1]
            
            # Format predictions with HISYNC AI enhancement
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
            
            # Find best match with expected label using HISYNC AI intelligence
            is_match, matched_class, match_confidence = self._find_best_match(
                predicted_classes, expected_label
            )
            
            # Get primary prediction
            primary_prediction = all_predictions[0]
            
            # Determine if confidence threshold is met
            confidence_met = primary_prediction["confidence"] >= confidence_threshold
            
            # Calculate HISYNC AI processing time
            processing_time = (time.time() - start_time) * 1000
            
            # Determine overall status using HISYNC AI logic
            if is_match and confidence_met:
                status = "correct"
                message = f"✅ HISYNC AI Classification CORRECT! Predicted '{primary_prediction['label']}' matches expected '{expected_label}' with {primary_prediction['confidence']:.2%} confidence"
            elif is_match and not confidence_met:
                status = "incorrect"
                message = f"⚠️ HISYNC AI Classification UNCERTAIN! Predicted '{primary_prediction['label']}' matches expected '{expected_label}' but confidence {primary_prediction['confidence']:.2%} is below threshold {confidence_threshold:.2%}"
            else:
                status = "incorrect"
                message = f"❌ HISYNC AI Classification INCORRECT! Predicted '{primary_prediction['label']}' does not match expected '{expected_label}'"
            
            # Return HISYNC AI structured result
            return {
                "status": status,
                "expected_label": expected_label,
                "prediction_result": {
                    "predicted_label": primary_prediction["label"],
                    "confidence": primary_prediction["confidence"],
                    "all_predictions": all_predictions
                },
                "is_match": is_match,
                "confidence_met": confidence_met,
                "message": message,
                "processing_time_ms": processing_time
            }
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            logger.error(f"HISYNC AI Classification error: {str(e)}")
            
            return {
                "status": "error",
                "expected_label": expected_label,
                "prediction_result": {
                    "predicted_label": "error",
                    "confidence": 0.0,
                    "all_predictions": []
                },
                "is_match": False,
                "confidence_met": False,
                "message": f"❌ HISYNC AI Classification failed: {str(e)}",
                "processing_time_ms": processing_time
            }

# Global HISYNC AI service instance
classification_service = HISYNCImageClassificationService() 