"""
Error Level Analysis (ELA) Feature Extraction
==============================================

Detects JPEG compression artifacts and inconsistencies that indicate
image manipulation or AI generation.
"""

import cv2
import numpy as np
from PIL import Image
import io
from pathlib import Path
from typing import Union, Optional, List, Tuple
from dataclasses import dataclass

try:
    from config import FEATURE_CONFIG
except ImportError:
    FEATURE_CONFIG = None

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class ELAFeatures:
    """Container for ELA-based features."""
    
    ela_mean: float
    ela_std: float
    ela_max: float
    ela_entropy: float
    ela_uniformity: float
    block_variance: float
    edge_ela_ratio: float
    texture_ela_ratio: float
    quality_estimates: np.ndarray
    regional_stats: np.ndarray
    
    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        base_features = np.array([
            self.ela_mean,
            self.ela_std,
            self.ela_max,
            self.ela_entropy,
            self.ela_uniformity,
            self.block_variance,
            self.edge_ela_ratio,
            self.texture_ela_ratio
        ])
        return np.concatenate([base_features, self.quality_estimates, self.regional_stats])


class ELAFeatureExtractor:
    """
    Error Level Analysis feature extractor.
    
    ELA works by resaving an image at a known quality level and comparing
    the difference. Areas that have been modified or generated differently
    will show different error levels.
    
    AI-generated images often show uniform ELA patterns, while authentic
    photos with edits show localized differences.
    
    Attributes
    ----------
    quality_levels : List[int]
        JPEG quality levels to test
    block_size : int
        Block size for regional analysis
    """
    
    def __init__(
        self,
        quality_levels: Optional[List[int]] = None,
        block_size: int = 8
    ):
        """
        Initialize the ELA feature extractor.
        
        Parameters
        ----------
        quality_levels : List[int], optional
            JPEG quality levels for multi-quality ELA
        block_size : int
            Block size for block-based analysis
        """
        if FEATURE_CONFIG:
            self.quality_levels = FEATURE_CONFIG.ela_quality_levels
            self.block_size = FEATURE_CONFIG.ela_block_size
        else:
            self.quality_levels = quality_levels or [90, 75, 50]
            self.block_size = block_size
        
        logger.debug(f"Initialized ELAFeatureExtractor with qualities={self.quality_levels}")
    
    @log_execution_time
    def extract(self, image: Union[str, Path, np.ndarray]) -> ELAFeatures:
        """
        Extract ELA-based forensic features from an image.
        
        Parameters
        ----------
        image : str, Path, or np.ndarray
            Input image path or array
            
        Returns
        -------
        ELAFeatures
            Comprehensive ELA feature set
        """
        # Load image
        if isinstance(image, (str, Path)):
            original = Image.open(str(image)).convert('RGB')
        else:
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original = Image.fromarray(image)
        
        original_array = np.array(original).astype(np.float64)
        
        # Compute ELA at multiple quality levels
        ela_maps = []
        quality_estimates = []
        
        for quality in self.quality_levels:
            ela_map = self._compute_ela(original, quality)
            ela_maps.append(ela_map)
            quality_estimates.append(self._estimate_quality(ela_map))
        
        # Use primary quality level for main features
        primary_ela = ela_maps[0]
        
        # Compute features
        features = self._compute_features(
            primary_ela, 
            ela_maps,
            original_array,
            np.array(quality_estimates)
        )
        
        return features
    
    def _compute_ela(self, image: Image.Image, quality: int) -> np.ndarray:
        """
        Compute ELA map for a given quality level.
        
        Parameters
        ----------
        image : PIL.Image
            Original image
        quality : int
            JPEG quality level (1-100)
            
        Returns
        -------
        np.ndarray
            ELA difference map
        """
        # Resave at specified quality
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        resaved = Image.open(buffer).convert('RGB')
        
        # Compute difference
        original_array = np.array(image).astype(np.float64)
        resaved_array = np.array(resaved).astype(np.float64)
        
        # ELA is the absolute difference, scaled
        ela = np.abs(original_array - resaved_array)
        
        # Convert to grayscale ELA
        ela_gray = np.mean(ela, axis=2)
        
        # Scale to enhance visibility
        scale = 255.0 / (np.max(ela_gray) + 1e-10)
        ela_scaled = ela_gray * scale
        
        return ela_scaled
    
    def _estimate_quality(self, ela_map: np.ndarray) -> float:
        """Estimate the original JPEG quality from ELA characteristics."""
        # Lower mean ELA suggests higher original quality
        mean_ela = np.mean(ela_map)
        # Normalize to 0-100 range (inverse relationship)
        estimated_quality = 100 - min(mean_ela * 2, 100)
        return float(estimated_quality)
    
    def _compute_features(
        self,
        primary_ela: np.ndarray,
        all_ela_maps: List[np.ndarray],
        original: np.ndarray,
        quality_estimates: np.ndarray
    ) -> ELAFeatures:
        """Compute all ELA-based features."""
        
        # Basic statistics
        ela_mean = np.mean(primary_ela)
        ela_std = np.std(primary_ela)
        ela_max = np.max(primary_ela)
        
        # ELA entropy
        ela_entropy = self._compute_entropy(primary_ela)
        
        # Uniformity (inverse of variance across blocks)
        ela_uniformity = self._compute_uniformity(primary_ela)
        
        # Block variance (for detecting JPEG block artifacts)
        block_variance = self._compute_block_variance(primary_ela)
        
        # Edge vs. smooth area ELA ratio
        edge_ela_ratio = self._compute_edge_ela_ratio(primary_ela, original)
        
        # Texture vs. flat area ELA ratio
        texture_ela_ratio = self._compute_texture_ela_ratio(primary_ela, original)
        
        # Regional statistics (divide image into regions)
        regional_stats = self._compute_regional_stats(primary_ela)
        
        return ELAFeatures(
            ela_mean=ela_mean,
            ela_std=ela_std,
            ela_max=ela_max,
            ela_entropy=ela_entropy,
            ela_uniformity=ela_uniformity,
            block_variance=block_variance,
            edge_ela_ratio=edge_ela_ratio,
            texture_ela_ratio=texture_ela_ratio,
            quality_estimates=quality_estimates,
            regional_stats=regional_stats
        )
    
    def _compute_entropy(self, ela_map: np.ndarray) -> float:
        """Compute Shannon entropy of ELA map."""
        # Normalize and compute histogram
        ela_norm = (ela_map - np.min(ela_map)) / (np.max(ela_map) - np.min(ela_map) + 1e-10)
        hist, _ = np.histogram(ela_norm, bins=256, range=(0, 1), density=True)
        
        # Remove zeros
        hist = hist[hist > 0]
        
        # Compute entropy
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        
        return float(entropy)
    
    def _compute_uniformity(self, ela_map: np.ndarray) -> float:
        """
        Compute uniformity measure.
        
        AI-generated images tend to have more uniform ELA patterns.
        """
        h, w = ela_map.shape
        block_h = h // self.block_size
        block_w = w // self.block_size
        
        block_means = []
        for i in range(self.block_size):
            for j in range(self.block_size):
                block = ela_map[
                    i * block_h:(i + 1) * block_h,
                    j * block_w:(j + 1) * block_w
                ]
                block_means.append(np.mean(block))
        
        # Uniformity is inverse of variance
        variance = np.var(block_means)
        uniformity = 1.0 / (1.0 + variance)
        
        return float(uniformity)
    
    def _compute_block_variance(self, ela_map: np.ndarray) -> float:
        """
        Compute variance within 8x8 JPEG blocks.
        
        JPEG artifacts create consistent patterns within blocks.
        """
        h, w = ela_map.shape
        block_size = 8  # JPEG block size
        
        variances = []
        for i in range(0, h - block_size, block_size):
            for j in range(0, w - block_size, block_size):
                block = ela_map[i:i+block_size, j:j+block_size]
                variances.append(np.var(block))
        
        return float(np.mean(variances))
    
    def _compute_edge_ela_ratio(
        self,
        ela_map: np.ndarray,
        original: np.ndarray
    ) -> float:
        """
        Compute ratio of ELA in edge regions vs. smooth regions.
        
        Real photos should have higher ELA near edges.
        """
        # Convert to grayscale for edge detection
        if len(original.shape) == 3:
            gray = np.mean(original, axis=2).astype(np.uint8)
        else:
            gray = original.astype(np.uint8)
        
        # Detect edges
        edges = cv2.Canny(gray, 50, 150)
        edge_mask = edges > 0
        
        # Dilate edges slightly
        kernel = np.ones((3, 3), np.uint8)
        edge_mask = cv2.dilate(edge_mask.astype(np.uint8), kernel, iterations=1) > 0
        
        # Resize mask to match ELA map
        if edge_mask.shape != ela_map.shape:
            edge_mask = cv2.resize(
                edge_mask.astype(np.uint8), 
                (ela_map.shape[1], ela_map.shape[0])
            ) > 0
        
        # Compute ratio
        edge_ela = np.mean(ela_map[edge_mask]) if np.any(edge_mask) else 0
        smooth_ela = np.mean(ela_map[~edge_mask]) if np.any(~edge_mask) else 0
        
        ratio = edge_ela / (smooth_ela + 1e-10)
        
        return float(ratio)
    
    def _compute_texture_ela_ratio(
        self,
        ela_map: np.ndarray,
        original: np.ndarray
    ) -> float:
        """
        Compute ratio of ELA in textured vs. flat regions.
        """
        # Convert to grayscale
        if len(original.shape) == 3:
            gray = np.mean(original, axis=2).astype(np.float64)
        else:
            gray = original.astype(np.float64)
        
        # Compute local variance as texture measure
        from scipy import ndimage
        local_mean = ndimage.uniform_filter(gray, size=11)
        local_sqr_mean = ndimage.uniform_filter(gray ** 2, size=11)
        local_var = local_sqr_mean - local_mean ** 2
        
        # Threshold for texture vs. flat
        texture_threshold = np.median(local_var)
        texture_mask = local_var > texture_threshold
        
        # Resize mask to match ELA map
        if texture_mask.shape != ela_map.shape:
            texture_mask = cv2.resize(
                texture_mask.astype(np.uint8),
                (ela_map.shape[1], ela_map.shape[0])
            ) > 0
        
        # Compute ratio
        texture_ela = np.mean(ela_map[texture_mask]) if np.any(texture_mask) else 0
        flat_ela = np.mean(ela_map[~texture_mask]) if np.any(~texture_mask) else 0
        
        ratio = texture_ela / (flat_ela + 1e-10)
        
        return float(ratio)
    
    def _compute_regional_stats(
        self,
        ela_map: np.ndarray,
        grid_size: int = 4
    ) -> np.ndarray:
        """
        Compute statistics for image regions.
        
        Helps detect localized manipulation.
        """
        h, w = ela_map.shape
        region_h = h // grid_size
        region_w = w // grid_size
        
        stats = []
        for i in range(grid_size):
            for j in range(grid_size):
                region = ela_map[
                    i * region_h:(i + 1) * region_h,
                    j * region_w:(j + 1) * region_w
                ]
                stats.extend([np.mean(region), np.std(region)])
        
        return np.array(stats)
    
    def visualize_ela(
        self,
        image: Union[str, Path, np.ndarray],
        quality: int = 90,
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Create ELA visualization.
        
        Parameters
        ----------
        image : str, Path, or np.ndarray
            Input image
        quality : int
            JPEG quality for ELA
        save_path : str or Path, optional
            Path to save visualization
            
        Returns
        -------
        np.ndarray
            ELA visualization
        """
        import matplotlib.pyplot as plt
        
        # Load image
        if isinstance(image, (str, Path)):
            original = Image.open(str(image)).convert('RGB')
        else:
            if len(image.shape) == 3:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            original = Image.fromarray(image)
        
        ela_map = self._compute_ela(original, quality)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(original)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # ELA map
        im = axes[1].imshow(ela_map, cmap='jet')
        axes[1].set_title(f'ELA (Quality={quality})')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)
        
        # Histogram
        axes[2].hist(ela_map.flatten(), bins=50, color='steelblue', alpha=0.7)
        axes[2].set_title('ELA Distribution')
        axes[2].set_xlabel('ELA Value')
        axes[2].set_ylabel('Frequency')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved ELA visualization to {save_path}")
        
        # Convert to array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_array


def extract_ela_features(image_path: Union[str, Path]) -> np.ndarray:
    """
    Convenience function to extract ELA features from an image.
    
    Parameters
    ----------
    image_path : str or Path
        Path to input image
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    extractor = ELAFeatureExtractor()
    features = extractor.extract(image_path)
    return features.to_array()
