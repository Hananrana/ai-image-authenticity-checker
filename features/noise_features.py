"""
Noise Pattern Analysis Features
===============================

PRNU and noise residual analysis for AI image detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Union, Optional
from dataclasses import dataclass
from scipy import ndimage

try:
    from config import FEATURE_CONFIG
except ImportError:
    FEATURE_CONFIG = None

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class NoiseFeatures:
    """Container for noise-based features."""
    noise_mean: float
    noise_std: float
    noise_entropy: float
    noise_uniformity: float
    local_variance_mean: float
    local_variance_std: float
    snr_estimate: float
    highfreq_noise_ratio: float
    patch_consistency: float
    
    def to_array(self) -> np.ndarray:
        return np.array([
            self.noise_mean, self.noise_std, self.noise_entropy,
            self.noise_uniformity, self.local_variance_mean,
            self.local_variance_std, self.snr_estimate,
            self.highfreq_noise_ratio, self.patch_consistency
        ])


class NoiseFeatureExtractor:
    """Noise pattern analyzer for image forensics."""
    
    def __init__(self, patch_size: int = 64, num_patches: int = 16):
        if FEATURE_CONFIG:
            self.patch_size = FEATURE_CONFIG.noise_patch_size
            self.num_patches = FEATURE_CONFIG.noise_num_patches
        else:
            self.patch_size = patch_size
            self.num_patches = num_patches
    
    @log_execution_time
    def extract(self, image: Union[str, Path, np.ndarray]) -> NoiseFeatures:
        """Extract noise features from an image."""
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            img = image.copy()
        
        img = cv2.resize(img, (256, 256))
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float64)
        else:
            gray = img.astype(np.float64)
        
        # Extract noise residual
        denoised = cv2.GaussianBlur(gray, (5, 5), 1.5)
        noise = gray - denoised
        
        # Statistics
        noise_mean = np.mean(np.abs(noise))
        noise_std = np.std(noise)
        
        # Entropy
        hist, _ = np.histogram(noise.flatten(), bins=64, density=True)
        hist_nz = hist[hist > 0]
        noise_entropy = -np.sum(hist_nz * np.log2(hist_nz + 1e-10))
        
        # Uniformity
        noise_uniformity = self._compute_uniformity(noise)
        
        # Local variance
        local_var = ndimage.uniform_filter(noise**2, 8) - ndimage.uniform_filter(noise, 8)**2
        local_variance_mean = np.mean(local_var)
        local_variance_std = np.std(local_var)
        
        # SNR
        signal_power = np.var(gray)
        noise_power = np.var(noise)
        snr_estimate = 10 * np.log10(signal_power / (noise_power + 1e-10))
        
        # High-freq noise
        highfreq_noise_ratio = self._compute_highfreq_ratio(noise)
        
        # Patch consistency
        patch_consistency = self._compute_patch_consistency(noise)
        
        return NoiseFeatures(
            noise_mean=float(noise_mean), noise_std=float(noise_std),
            noise_entropy=float(noise_entropy), noise_uniformity=float(noise_uniformity),
            local_variance_mean=float(local_variance_mean),
            local_variance_std=float(local_variance_std),
            snr_estimate=float(snr_estimate),
            highfreq_noise_ratio=float(highfreq_noise_ratio),
            patch_consistency=float(patch_consistency)
        )
    
    def _compute_uniformity(self, noise: np.ndarray) -> float:
        h, w = noise.shape
        block_size = 8
        block_h, block_w = h // block_size, w // block_size
        block_vars = []
        for i in range(block_size):
            for j in range(block_size):
                block = noise[i*block_h:(i+1)*block_h, j*block_w:(j+1)*block_w]
                block_vars.append(np.var(block))
        return 1.0 / (1.0 + np.var(block_vars))
    
    def _compute_highfreq_ratio(self, noise: np.ndarray) -> float:
        from scipy.fft import fft2, fftshift
        fft = fftshift(fft2(noise))
        mag = np.abs(fft)
        h, w = mag.shape
        ch, cw = h // 2, w // 2
        total = np.sum(mag)
        center = np.sum(mag[ch-32:ch+32, cw-32:cw+32])
        return (total - center) / (total + 1e-10)
    
    def _compute_patch_consistency(self, noise: np.ndarray) -> float:
        h, w = noise.shape
        patches = []
        for _ in range(self.num_patches):
            y = np.random.randint(0, h - self.patch_size)
            x = np.random.randint(0, w - self.patch_size)
            patch = noise[y:y+self.patch_size, x:x+self.patch_size]
            patches.append(np.std(patch))
        return 1.0 / (1.0 + np.std(patches))


def extract_noise_features(image_path: Union[str, Path]) -> np.ndarray:
    """Extract noise features from an image."""
    return NoiseFeatureExtractor().extract(image_path).to_array()
