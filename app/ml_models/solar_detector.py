"""
Solar Panel Detection ML Model
"""
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import requests
import base64
from io import BytesIO
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

class SimpleSolarDetector(nn.Module):
    """
    Simple CNN-based solar panel detector
    This is a lightweight model for detection
    """
    def __init__(self, num_classes=2):
        super(SimpleSolarDetector, self).__init__()
        # Simple feature extractor
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(64, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

class SolarPanelDetector:
    """
    Main detector class that wraps the ML model
    """
    def __init__(self, model_path: Optional[str] = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.model_loaded = False
        self.model_path = model_path
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def load_model(self):
        """Load the model (lazy loading)"""
        if self.model_loaded:
            return
        
        try:
            if self.model_path and os.path.exists(self.model_path):
                # Load from file
                self.model = SimpleSolarDetector()
                self.model.load_state_dict(torch.load(self.model_path, map_location=self.device))
                logger.info(f"Loaded model from {self.model_path}")
            else:
                # Initialize with random weights (for demo - in production, use trained weights)
                self.model = SimpleSolarDetector()
                logger.warning("Using untrained model with random weights")
            
            self.model.to(self.device)
            self.model.eval()
            self.model_loaded = True
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Fallback: use a simple rule-based detector
            self.model = None
            self.model_loaded = False
    
    def detect_from_image(self, image: Image.Image) -> Dict:
        """
        Detect solar panels from a PIL Image
        Returns detection results
        """
        if not self.model_loaded:
            self.load_model()
        
        try:
            # Preprocess image
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            if self.model:
                # Run inference
                with torch.no_grad():
                    outputs = self.model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    confidence, predicted = torch.max(probabilities, 1)
                    
                    has_solar = predicted.item() == 1
                    conf_score = confidence.item()
            else:
                # Fallback: analyze image for blue/dark rectangular shapes
                has_solar, conf_score = self._rule_based_detection(image)
            
            # Estimate panel count and area (simplified)
            if has_solar:
                panel_count = max(1, int(conf_score * 20))  # Rough estimate
                pv_area_sqm = panel_count * 1.6  # Average panel is ~1.6 m²
                capacity_kw = pv_area_sqm * 0.15  # ~150W per m²
            else:
                panel_count = 0
                pv_area_sqm = 0.0
                capacity_kw = 0.0
            
            return {
                "has_solar": has_solar,
                "confidence": float(conf_score),
                "panel_count": panel_count,
                "pv_area_sqm": float(pv_area_sqm),
                "capacity_kw": float(capacity_kw),
                "predictions": [{
                    "class": "solar_panel" if has_solar else "no_solar",
                    "confidence": float(conf_score),
                    "bbox": [0, 0, image.width, image.height]  # Full image bbox
                }]
            }
        except Exception as e:
            logger.error(f"Detection error: {e}")
            return {
                "has_solar": False,
                "confidence": 0.0,
                "panel_count": 0,
                "pv_area_sqm": 0.0,
                "capacity_kw": 0.0,
                "error": str(e)
            }
    
    def detect_from_url(self, image_url: str) -> Dict:
        """Detect solar panels from an image URL"""
        try:
            response = requests.get(image_url, timeout=15)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGB")
            return self.detect_from_image(image)
        except Exception as e:
            logger.error(f"Failed to fetch image from URL: {e}")
            return {
                "has_solar": False,
                "confidence": 0.0,
                "panel_count": 0,
                "pv_area_sqm": 0.0,
                "capacity_kw": 0.0,
                "error": str(e)
            }
    
    def _rule_based_detection(self, image: Image.Image) -> Tuple[bool, float]:
        """
        Simple rule-based detection as fallback
        Looks for blue/dark rectangular regions (typical of solar panels)
        """
        try:
            # Convert to numpy array
            img_array = np.array(image)
            
            # Convert to HSV for better color detection
            from PIL import Image as PILImage
            import cv2
            if hasattr(cv2, 'cvtColor'):
                hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
                # Look for blue/dark regions (solar panels are often dark blue/black)
                lower_blue = np.array([100, 50, 50])
                upper_blue = np.array([130, 255, 255])
                mask = cv2.inRange(hsv, lower_blue, upper_blue)
                blue_ratio = np.sum(mask > 0) / mask.size
                
                # Also check for dark regions
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                dark_ratio = np.sum(gray < 50) / gray.size
                
                # Combine indicators
                solar_indicator = (blue_ratio + dark_ratio * 0.5) / 1.5
                has_solar = solar_indicator > 0.1
                confidence = min(0.9, solar_indicator * 2)
                
                return has_solar, confidence
            else:
                # Fallback: just check image size and assume no solar if too small
                return False, 0.1
        except Exception:
            return False, 0.1

# Global detector instance (lazy loaded)
_detector_instance: Optional[SolarPanelDetector] = None

def get_detector() -> SolarPanelDetector:
    """Get or create the global detector instance"""
    global _detector_instance
    if _detector_instance is None:
        model_path = os.getenv("ML_MODEL_PATH", "model_weights/solar_detector.pth")
        _detector_instance = SolarPanelDetector(model_path=model_path)
    return _detector_instance

