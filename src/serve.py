"""
FastAPI service for DirtyCar binary classification.
Provides REST endpoints for car cleanliness prediction.
"""

import os
import json
import asyncio
import logging
from pathlib import Path
from typing import Dict, Optional, Union
from io import BytesIO

import numpy as np
import cv2
from PIL import Image
import requests
import onnxruntime as ort

from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, HttpUrl, validator
import uvicorn


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class URLRequest(BaseModel):
    """Request model for URL-based prediction."""
    url: HttpUrl
    
    @validator('url')
    def validate_url(cls, v):
        """Validate URL format."""
        url_str = str(v)
        if not any(url_str.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']):
            logger.warning(f"URL might not be an image: {url_str}")
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    label: str
    p_clean: float
    threshold: Optional[float] = None
    confidence: Optional[str] = None
    processing_time_ms: Optional[float] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    model_loaded: bool
    model_path: Optional[str] = None
    threshold_loaded: bool = False
    threshold_value: Optional[float] = None


class DirtyCarPredictor:
    """ONNX-based predictor for car cleanliness classification."""
    
    def __init__(self):
        # Environment variables
        self.model_path = os.getenv('MODEL_PATH', './artifacts/best.onnx')
        self.img_size = int(os.getenv('IMGSZ', '256'))
        self.manual_threshold = self._parse_float_env('THRESH')
        self.t_low = self._parse_float_env('T_LOW')
        self.t_high = self._parse_float_env('T_HIGH')
        
        # Model components
        self.session = None
        self.input_name = None
        self.output_name = None
        self.threshold_info = None
        self.class_names = ['clean', 'dirty']
        
        # Image preprocessing parameters (ImageNet normalization)
        self.mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        
        # Load model and threshold
        self._load_model()
        self._load_threshold()
        
        logger.info("DirtyCarPredictor initialized successfully")
    
    def _parse_float_env(self, env_var: str) -> Optional[float]:
        """Parse float from environment variable."""
        value = os.getenv(env_var)
        if value:
            try:
                return float(value)
            except ValueError:
                logger.warning(f"Invalid float value for {env_var}: {value}")
        return None
    
    def _load_model(self):
        """Load ONNX model."""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        logger.info(f"Loading ONNX model from {self.model_path}")
        
        # Setup providers (prefer CUDA if available)
        providers = []
        if ort.get_device() == 'GPU':
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        
        try:
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            # Get input/output names
            self.input_name = self.session.get_inputs()[0].name
            self.output_name = self.session.get_outputs()[0].name
            
            # Log model info
            input_shape = self.session.get_inputs()[0].shape
            logger.info(f"Model loaded successfully")
            logger.info(f"  Input shape: {input_shape}")
            logger.info(f"  Providers: {self.session.get_providers()}")
            
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            raise
    
    def _load_threshold(self):
        """Load threshold configuration."""
        threshold_path = Path(self.model_path).parent / 'threshold.json'
        
        if threshold_path.exists():
            try:
                with open(threshold_path, 'r') as f:
                    data = json.load(f)
                    
                # Extract threshold info from nested structure
                if 'threshold_selection' in data:
                    self.threshold_info = data['threshold_selection']
                else:
                    self.threshold_info = data
                
                logger.info(f"Threshold configuration loaded from {threshold_path}")
                logger.info(f"  T_clean: {self.threshold_info.get('T_clean', 'N/A')}")
                logger.info(f"  Target precision: {self.threshold_info.get('target_precision_clean', 'N/A')}")
                
            except Exception as e:
                logger.warning(f"Failed to load threshold config: {e}")
                self.threshold_info = None
        else:
            logger.warning(f"Threshold config not found: {threshold_path}")
            self.threshold_info = None
    
    def _get_threshold(self) -> float:
        """Get threshold value with priority: manual > config file > default."""
        if self.manual_threshold is not None:
            return self.manual_threshold
        
        if self.threshold_info and 'T_clean' in self.threshold_info:
            return self.threshold_info['T_clean']
        
        # Default threshold
        return 0.5
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for model inference.
        
        Args:
            image: Input image as numpy array (H, W, C) in BGR format
        
        Returns:
            Preprocessed image tensor (1, C, H, W)
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Resize
        image = cv2.resize(image, (self.img_size, self.img_size))
        
        # Convert to float and normalize to [0, 1]
        image = image.astype(np.float32) / 255.0
        
        # Apply ImageNet normalization
        image = (image - self.mean) / self.std
        
        # Convert HWC to CHW and add batch dimension
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, axis=0)
        
        return image
    
    def _postprocess_output(self, logits: np.ndarray) -> Dict[str, float]:
        """
        Postprocess model output to get probabilities.
        
        Args:
            logits: Raw model output logits
        
        Returns:
            Dictionary with probabilities and predictions
        """
        # Apply temperature scaling if available
        temperature = 1.0
        if self.threshold_info and 'temperature' in self.threshold_info:
            temperature = self.threshold_info['temperature']
        
        scaled_logits = logits / temperature
        
        # Softmax to get probabilities
        exp_logits = np.exp(scaled_logits - np.max(scaled_logits, axis=1, keepdims=True))
        probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Extract clean probability (class 0)
        p_clean = float(probabilities[0, 0])
        p_dirty = float(probabilities[0, 1])
        
        return {
            'p_clean': p_clean,
            'p_dirty': p_dirty,
            'probabilities': probabilities[0].tolist()
        }
    
    def _apply_business_logic(self, p_clean: float) -> Dict[str, Union[str, float]]:
        """
        Apply business logic to determine final prediction.
        
        Args:
            p_clean: Probability of clean class
        
        Returns:
            Dictionary with label and confidence information
        """
        threshold = self._get_threshold()
        
        # Check for uncertainty zone if T_LOW and T_HIGH are defined
        if self.t_low is not None and self.t_high is not None and self.t_low < self.t_high:
            if p_clean >= self.t_high:
                label = 'clean'
                confidence = 'high'
            elif p_clean <= self.t_low:
                label = 'dirty'
                confidence = 'high'
            else:
                label = 'unsure'  # Can be treated as dirty in business logic
                confidence = 'low'
        else:
            # Standard business rule: if p_clean >= threshold -> clean, else -> dirty
            if p_clean >= threshold:
                label = 'clean'
                confidence = 'high' if p_clean >= 0.8 else 'medium'
            else:
                label = 'dirty'
                confidence = 'high' if p_clean <= 0.2 else 'medium'
        
        return {
            'label': label,
            'confidence': confidence,
            'threshold': threshold
        }
    
    def predict_image(self, image: np.ndarray) -> Dict[str, Union[str, float]]:
        """
        Predict car cleanliness from image.
        
        Args:
            image: Input image as numpy array
        
        Returns:
            Prediction results
        """
        import time
        start_time = time.time()
        
        try:
            # Preprocess
            processed_image = self._preprocess_image(image)
            
            # Inference
            logits = self.session.run([self.output_name], {self.input_name: processed_image})[0]
            
            # Postprocess
            prob_results = self._postprocess_output(logits)
            
            # Apply business logic
            business_results = self._apply_business_logic(prob_results['p_clean'])
            
            # Combine results
            result = {
                'label': business_results['label'],
                'p_clean': prob_results['p_clean'],
                'threshold': business_results['threshold'],
                'confidence': business_results['confidence'],
                'processing_time_ms': (time.time() - start_time) * 1000
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
    
    def is_healthy(self) -> Dict[str, Union[str, bool, float]]:
        """Check if the predictor is healthy."""
        return {
            'status': 'ok' if self.session is not None else 'error',
            'model_loaded': self.session is not None,
            'model_path': self.model_path,
            'threshold_loaded': self.threshold_info is not None,
            'threshold_value': self._get_threshold()
        }


# Global predictor instance
predictor = None


def get_predictor() -> DirtyCarPredictor:
    """Dependency to get predictor instance."""
    global predictor
    if predictor is None:
        predictor = DirtyCarPredictor()
    return predictor


# FastAPI app
app = FastAPI(
    title="DirtyCar Classifier API",
    description="Binary classification API for car cleanliness detection (clean vs dirty)",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize predictor on startup."""
    global predictor
    try:
        predictor = DirtyCarPredictor()
        logger.info("API startup completed successfully")
    except Exception as e:
        logger.error(f"Failed to initialize predictor: {e}")
        raise


@app.get("/healthz", response_model=HealthResponse)
async def health_check(pred: DirtyCarPredictor = Depends(get_predictor)):
    """Health check endpoint."""
    health_info = pred.is_healthy()
    return HealthResponse(**health_info)


@app.post("/predict/file", response_model=PredictionResponse)
async def predict_file(
    file: UploadFile = File(...),
    pred: DirtyCarPredictor = Depends(get_predictor)
):
    """
    Predict car cleanliness from uploaded image file.
    
    Args:
        file: Image file (JPEG, PNG, etc.)
    
    Returns:
        Prediction results with label and confidence
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image
        contents = await file.read()
        
        # Convert to numpy array
        image = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Predict
        result = pred.predict_image(image)
        
        return PredictionResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/url", response_model=PredictionResponse)
async def predict_url(
    request: URLRequest,
    pred: DirtyCarPredictor = Depends(get_predictor)
):
    """
    Predict car cleanliness from image URL.
    
    Args:
        request: URL request containing image URL
    
    Returns:
        Prediction results with label and confidence
    """
    try:
        # Download image
        response = requests.get(str(request.url), timeout=10)
        response.raise_for_status()
        
        # Convert to numpy array
        image = np.frombuffer(response.content, np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image URL or unsupported format")
        
        # Predict
        result = pred.predict_image(image)
        
        return PredictionResponse(**result)
        
    except requests.RequestException as e:
        logger.error(f"Failed to download image from URL: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download image: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"URL prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "DirtyCar Classifier API",
        "version": "1.0.0",
        "description": "Binary classification for car cleanliness detection",
        "endpoints": {
            "health": "/healthz",
            "predict_file": "/predict/file",
            "predict_url": "/predict/url",
            "docs": "/docs"
        },
        "business_rule": "if p_clean >= threshold then 'clean' else 'dirty'"
    }


@app.get("/model/info")
async def model_info(pred: DirtyCarPredictor = Depends(get_predictor)):
    """Get model information and configuration."""
    health = pred.is_healthy()
    
    info = {
        "model_path": pred.model_path,
        "image_size": pred.img_size,
        "class_names": pred.class_names,
        "threshold": pred._get_threshold(),
        "threshold_source": "manual" if pred.manual_threshold is not None else "config" if pred.threshold_info else "default",
        "uncertainty_zone": {
            "enabled": pred.t_low is not None and pred.t_high is not None,
            "t_low": pred.t_low,
            "t_high": pred.t_high
        },
        "model_loaded": health["model_loaded"],
        "threshold_loaded": health["threshold_loaded"]
    }
    
    if pred.threshold_info:
        info["threshold_config"] = pred.threshold_info
    
    return info


if __name__ == "__main__":
    # Configuration from environment
    host = os.getenv('API_HOST', '0.0.0.0')
    port = int(os.getenv('API_PORT', '8000'))
    
    # Run server
    uvicorn.run(
        "serve:app",
        host=host,
        port=port,
        reload=False,
        workers=1,
        log_level="info"
    )
