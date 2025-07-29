from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, List, Dict, Any
from enum import Enum

# Configure Pydantic to avoid namespace warnings
class BaseModelConfig(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

class ClassificationStatus(str, Enum):
    CORRECT = "correct"
    INCORRECT = "incorrect"
    ERROR = "error"

class ImageClassificationRequest(BaseModelConfig):
    expected_label: str = Field(..., description="The expected label for the image")
    confidence_threshold: float = Field(default=0.8, ge=0.0, le=1.0, description="Minimum confidence threshold")

class PredictionResult(BaseModelConfig):
    predicted_label: str = Field(..., description="The predicted label from HISYNC AI model")
    confidence: float = Field(..., ge=0.0, le=1.0, description="HISYNC AI confidence score of the prediction")
    all_predictions: List[Dict[str, Any]] = Field(..., description="All top predictions with confidence scores from HISYNC AI")

class ClassificationResponse(BaseModelConfig):
    status: ClassificationStatus = Field(..., description="Whether the HISYNC AI classification is correct or incorrect")
    expected_label: str = Field(..., description="The expected label provided by user")
    prediction_result: PredictionResult = Field(..., description="HISYNC AI model prediction results")
    is_match: bool = Field(..., description="Whether HISYNC AI predicted label matches expected label")
    confidence_met: bool = Field(..., description="Whether HISYNC AI confidence threshold was met")
    message: str = Field(..., description="Human readable message about the HISYNC AI result")
    processing_time_ms: float = Field(..., description="Time taken by HISYNC AI to process the image")

class ErrorResponse(BaseModelConfig):
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Detailed error message from HISYNC AI system")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    support: Optional[str] = Field(default="Contact support@hisync.in for assistance", description="HISYNC support information")

class HealthResponse(BaseModelConfig):
    status: str = Field(..., description="HISYNC AI system health status")
    ai_model_loaded: bool = Field(..., description="Whether HISYNC AI model is loaded and ready", alias="model_loaded")
    version: str = Field(..., description="HISYNC AI API version")
    supported_formats: List[str] = Field(..., description="Image formats supported by HISYNC AI")
    resnet_available: Optional[bool] = Field(default=False, description="Whether Google ResNet v2 model is available")
    yolo12_available: Optional[bool] = Field(default=False, description="Whether YOLO12 model is available")
    
    class Config:
        populate_by_name = True  # Allow both field name and alias 