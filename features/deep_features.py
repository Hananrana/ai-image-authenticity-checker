"""
Deep Learning Feature Extraction
=================================

CNN-based feature extraction using pretrained models.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass

try:
    import torch
    import torch.nn as nn
    from torchvision import transforms
    from PIL import Image
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from config import FEATURE_CONFIG, DL_CONFIG
except ImportError:
    FEATURE_CONFIG = None
    DL_CONFIG = None

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class DeepFeatures:
    """Container for deep learning features."""
    features: np.ndarray
    
    def to_array(self) -> np.ndarray:
        return self.features


class DeepFeatureExtractor:
    """CNN-based feature extractor using pretrained EfficientNet."""
    
    def __init__(self, model_name: str = "efficientnet_b0", device: Optional[str] = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for deep features. Install with: pip install torch torchvision")
        
        self.model_name = model_name if not FEATURE_CONFIG else FEATURE_CONFIG.deep_model_name
        self.device = device or (DL_CONFIG.device if DL_CONFIG else "cuda")
        
        if self.device == "cuda" and not torch.cuda.is_available():
            self.device = "cpu"
            logger.warning("CUDA not available, using CPU")
        
        self.model = self._load_model()
        self.transform = self._get_transform()
        logger.debug(f"Initialized DeepFeatureExtractor with {self.model_name} on {self.device}")
    
    def _load_model(self) -> nn.Module:
        """Load pretrained model."""
        try:
            import timm
            model = timm.create_model(self.model_name, pretrained=True, num_classes=0)
        except ImportError:
            from torchvision import models
            if "efficientnet" in self.model_name:
                model = models.efficientnet_b0(pretrained=True)
                model.classifier = nn.Identity()
            else:
                model = models.resnet50(pretrained=True)
                model.fc = nn.Identity()
        
        model = model.to(self.device)
        model.eval()
        return model
    
    def _get_transform(self) -> transforms.Compose:
        """Get preprocessing transform."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    @log_execution_time
    def extract(self, image: Union[str, Path, np.ndarray]) -> DeepFeatures:
        """Extract deep features from an image."""
        if isinstance(image, (str, Path)):
            img = Image.open(str(image)).convert('RGB')
        elif isinstance(image, np.ndarray):
            if len(image.shape) == 2:
                image = np.stack([image] * 3, axis=-1)
            img = Image.fromarray(image)
        else:
            raise ValueError("Invalid image type")
        
        tensor = self.transform(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(tensor)
        
        features_np = features.cpu().numpy().flatten()
        return DeepFeatures(features=features_np)
    
    def extract_batch(self, images: list) -> np.ndarray:
        """Extract features from multiple images."""
        tensors = []
        for img in images:
            if isinstance(img, (str, Path)):
                pil_img = Image.open(str(img)).convert('RGB')
            elif isinstance(img, np.ndarray):
                pil_img = Image.fromarray(img if len(img.shape) == 3 else np.stack([img]*3, -1))
            else:
                continue
            tensors.append(self.transform(pil_img))
        
        if not tensors:
            return np.array([])
        
        batch = torch.stack(tensors).to(self.device)
        
        with torch.no_grad():
            features = self.model(batch)
        
        return features.cpu().numpy()


def extract_deep_features(image_path: Union[str, Path]) -> np.ndarray:
    """Extract deep features from an image."""
    if not TORCH_AVAILABLE:
        logger.warning("PyTorch not available, returning empty features")
        return np.zeros(1280)  # EfficientNet-B0 feature size
    return DeepFeatureExtractor().extract(image_path).to_array()
