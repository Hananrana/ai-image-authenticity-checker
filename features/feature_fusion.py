"""
Feature Fusion Module
=====================

Combines features from all extractors into a unified representation.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
import joblib

try:
    from config import PROJECT_ROOT, MODEL_DIR
except ImportError:
    PROJECT_ROOT = Path(__file__).parent.parent
    MODEL_DIR = PROJECT_ROOT / "saved_models"

from .fft_features import FFTFeatureExtractor, extract_fft_features
from .ela_features import ELAFeatureExtractor, extract_ela_features
from .texture_features import TextureFeatureExtractor, extract_texture_features
from .noise_features import NoiseFeatureExtractor, extract_noise_features

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class FusedFeatures:
    """Container for fused features."""
    fft_features: np.ndarray
    ela_features: np.ndarray
    texture_features: np.ndarray
    noise_features: np.ndarray
    deep_features: Optional[np.ndarray] = None
    
    def to_array(self, include_deep: bool = False) -> np.ndarray:
        """Concatenate all features."""
        features = [self.fft_features, self.ela_features, self.texture_features, self.noise_features]
        if include_deep and self.deep_features is not None:
            features.append(self.deep_features)
        return np.concatenate(features)


class FeatureFusion:
    """
    Unified feature extraction pipeline.
    
    Combines FFT, ELA, texture, noise, and optionally deep features
    into a single normalized feature vector.
    """
    
    def __init__(self, include_deep: bool = False, scaler_path: Optional[Path] = None):
        self.include_deep = include_deep
        
        # Initialize extractors
        self.fft_extractor = FFTFeatureExtractor()
        self.ela_extractor = ELAFeatureExtractor()
        self.texture_extractor = TextureFeatureExtractor()
        self.noise_extractor = NoiseFeatureExtractor()
        
        if include_deep:
            try:
                from .deep_features import DeepFeatureExtractor
                self.deep_extractor = DeepFeatureExtractor()
            except ImportError:
                logger.warning("Deep features unavailable, disabling")
                self.include_deep = False
                self.deep_extractor = None
        else:
            self.deep_extractor = None
        
        # Scaler for normalization
        self.scaler = None
        if scaler_path and Path(scaler_path).exists():
            self.scaler = joblib.load(scaler_path)
            logger.info(f"Loaded scaler from {scaler_path}")
    
    @log_execution_time
    def extract(self, image: Union[str, Path, np.ndarray]) -> FusedFeatures:
        """Extract all features from an image."""
        fft_feat = self.fft_extractor.extract(image).to_array()
        ela_feat = self.ela_extractor.extract(image).to_array()
        texture_feat = self.texture_extractor.extract(image).to_array()
        noise_feat = self.noise_extractor.extract(image).to_array()
        
        deep_feat = None
        if self.include_deep and self.deep_extractor:
            deep_feat = self.deep_extractor.extract(image).to_array()
        
        return FusedFeatures(
            fft_features=fft_feat,
            ela_features=ela_feat,
            texture_features=texture_feat,
            noise_features=noise_feat,
            deep_features=deep_feat
        )
    
    def extract_normalized(self, image: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Extract and normalize features."""
        features = self.extract(image).to_array(self.include_deep)
        
        if self.scaler:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        return features
    
    def fit_scaler(self, features: np.ndarray, save_path: Optional[Path] = None) -> StandardScaler:
        """Fit scaler on training features."""
        self.scaler = StandardScaler()
        self.scaler.fit(features)
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(self.scaler, save_path)
            logger.info(f"Saved scaler to {save_path}")
        
        return self.scaler
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for interpretability."""
        names = []
        
        # FFT features
        names.extend([f"fft_{i}" for i in range(48)])  # Approximate count
        
        # ELA features
        names.extend([f"ela_{i}" for i in range(43)])
        
        # Texture features
        names.extend([f"texture_{i}" for i in range(107)])
        
        # Noise features
        names.extend([f"noise_{i}" for i in range(9)])
        
        if self.include_deep:
            names.extend([f"deep_{i}" for i in range(1280)])
        
        return names


def extract_all_features(
    image_path: Union[str, Path],
    include_deep: bool = False,
    normalize: bool = False
) -> np.ndarray:
    """
    Convenience function to extract all features from an image.
    
    Parameters
    ----------
    image_path : str or Path
        Path to input image
    include_deep : bool
        Whether to include deep learning features
    normalize : bool
        Whether to normalize features (requires fitted scaler)
    
    Returns
    -------
    np.ndarray
        Combined feature vector
    """
    fusion = FeatureFusion(include_deep=include_deep)
    
    if normalize:
        return fusion.extract_normalized(image_path)
    else:
        return fusion.extract(image_path).to_array(include_deep)
