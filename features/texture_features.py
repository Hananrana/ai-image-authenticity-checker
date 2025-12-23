"""
Texture Feature Extraction
==========================

LBP and GLCM features for detecting AI-generated image artifacts.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional, List
from dataclasses import dataclass
from skimage.feature import local_binary_pattern, graycomatrix, graycoprops

try:
    from config import FEATURE_CONFIG
except ImportError:
    FEATURE_CONFIG = None

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class TextureFeatures:
    """Container for texture-based features."""
    lbp_histogram: np.ndarray
    lbp_mean: float
    lbp_variance: float
    lbp_entropy: float
    glcm_contrast: np.ndarray
    glcm_homogeneity: np.ndarray
    glcm_energy: np.ndarray
    glcm_correlation: np.ndarray
    gabor_means: np.ndarray
    gabor_stds: np.ndarray
    
    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        scalar = np.array([self.lbp_mean, self.lbp_variance, self.lbp_entropy])
        glcm = np.concatenate([
            self.glcm_contrast.flatten(), self.glcm_homogeneity.flatten(),
            self.glcm_energy.flatten(), self.glcm_correlation.flatten()
        ])
        return np.concatenate([scalar, self.lbp_histogram, glcm, self.gabor_means, self.gabor_stds])


class TextureFeatureExtractor:
    """Texture feature extractor using LBP, GLCM, and Gabor filters."""
    
    def __init__(self, lbp_radius: int = 3, lbp_n_points: int = 24):
        if FEATURE_CONFIG:
            self.lbp_radius = FEATURE_CONFIG.lbp_radius
            self.lbp_n_points = FEATURE_CONFIG.lbp_n_points
            self.glcm_distances = FEATURE_CONFIG.glcm_distances
            self.glcm_angles = FEATURE_CONFIG.glcm_angles
        else:
            self.lbp_radius = lbp_radius
            self.lbp_n_points = lbp_n_points
            self.glcm_distances = [1, 2, 5]
            self.glcm_angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        self.gabor_filters = self._build_gabor_bank()
    
    def _build_gabor_bank(self) -> List[np.ndarray]:
        """Build Gabor filter bank."""
        filters = []
        for freq in [0.1, 0.2, 0.3, 0.4]:
            for theta in [0, np.pi/4, np.pi/2, 3*np.pi/4]:
                kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 1.0/freq, 0.5, 0, cv2.CV_64F)
                filters.append(kernel)
        return filters
    
    @log_execution_time
    def extract(self, image: Union[str, Path, np.ndarray]) -> TextureFeatures:
        """Extract texture features from an image."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
        
        img = cv2.resize(img, (256, 256))
        
        # LBP
        lbp = local_binary_pattern(img, self.lbp_n_points, self.lbp_radius, 'uniform')
        n_bins = self.lbp_n_points + 2
        hist, _ = np.histogram(lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True)
        hist_nz = hist[hist > 0]
        lbp_entropy = -np.sum(hist_nz * np.log2(hist_nz + 1e-10))
        
        # GLCM
        img_q = (img // 4).astype(np.uint8)
        glcm = graycomatrix(img_q, self.glcm_distances, self.glcm_angles, 64, True, True)
        
        # Gabor
        means, stds = [], []
        for kernel in self.gabor_filters:
            resp = cv2.filter2D(img.astype(np.float64), cv2.CV_64F, kernel)
            means.append(np.mean(np.abs(resp)))
            stds.append(np.std(resp))
        
        return TextureFeatures(
            lbp_histogram=hist, lbp_mean=float(np.mean(lbp)),
            lbp_variance=float(np.var(lbp)), lbp_entropy=float(lbp_entropy),
            glcm_contrast=graycoprops(glcm, 'contrast'),
            glcm_homogeneity=graycoprops(glcm, 'homogeneity'),
            glcm_energy=graycoprops(glcm, 'energy'),
            glcm_correlation=graycoprops(glcm, 'correlation'),
            gabor_means=np.array(means), gabor_stds=np.array(stds)
        )


def extract_texture_features(image_path: Union[str, Path]) -> np.ndarray:
    """Extract texture features from an image."""
    return TextureFeatureExtractor().extract(image_path).to_array()
