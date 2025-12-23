"""
Enhanced FFT-Based Forensic Feature Extraction
==============================================

Advanced frequency domain analysis for detecting AI-generated images.
Includes GAN fingerprint detection, spectral band analysis, and 
azimuthally averaged power spectrum features.
"""

import cv2
import numpy as np
from scipy.fft import fft2, fftshift
from scipy import ndimage
from typing import Dict, List, Tuple, Union, Optional
from pathlib import Path
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
class FFTFeatures:
    """Container for FFT-based features."""
    
    low_freq_energy: float
    high_freq_energy: float
    spectral_ratio: float
    spectral_entropy: float
    azimuthal_profile: np.ndarray
    band_energies: np.ndarray
    spectral_flatness: float
    spectral_centroid: float
    spectral_rolloff: float
    gan_fingerprint_score: float
    
    def to_array(self) -> np.ndarray:
        """Convert to feature vector."""
        base_features = np.array([
            self.low_freq_energy,
            self.high_freq_energy,
            self.spectral_ratio,
            self.spectral_entropy,
            self.spectral_flatness,
            self.spectral_centroid,
            self.spectral_rolloff,
            self.gan_fingerprint_score
        ])
        return np.concatenate([base_features, self.azimuthal_profile, self.band_energies])


class FFTFeatureExtractor:
    """
    Advanced FFT-based feature extractor for image forensics.
    
    This extractor analyzes the frequency domain characteristics of images
    to detect artifacts typical of AI-generated content, including:
    - GAN fingerprints in periodic frequency patterns
    - Spectral distribution anomalies
    - Band energy ratios
    
    Attributes
    ----------
    image_size : int
        Standard size for frequency analysis
    low_freq_radius : int
        Radius defining low-frequency region
    num_bands : int
        Number of frequency bands for band analysis
    """
    
    def __init__(
        self,
        image_size: int = 256,
        low_freq_radius: int = 32,
        num_bands: int = 8
    ):
        """
        Initialize the FFT feature extractor.
        
        Parameters
        ----------
        image_size : int
            Size to resize images to before FFT analysis
        low_freq_radius : int
            Radius of the central low-frequency region
        num_bands : int
            Number of concentric frequency bands to analyze
        """
        if FEATURE_CONFIG:
            self.image_size = FEATURE_CONFIG.fft_image_size
            self.low_freq_radius = FEATURE_CONFIG.fft_low_freq_radius
            self.num_bands = FEATURE_CONFIG.fft_num_bands
        else:
            self.image_size = image_size
            self.low_freq_radius = low_freq_radius
            self.num_bands = num_bands
        
        # Pre-compute frequency masks
        self._init_masks()
        logger.debug(f"Initialized FFTFeatureExtractor with size={self.image_size}")
    
    def _init_masks(self):
        """Initialize frequency region masks."""
        h, w = self.image_size, self.image_size
        center_h, center_w = h // 2, w // 2
        
        # Create distance map from center
        y, x = np.ogrid[:h, :w]
        self.distance_map = np.sqrt((y - center_h)**2 + (x - center_w)**2)
        
        # Low frequency mask
        self.low_freq_mask = self.distance_map <= self.low_freq_radius
        
        # High frequency mask
        self.high_freq_mask = self.distance_map > self.low_freq_radius
        
        # Band masks for multi-scale analysis
        max_radius = np.sqrt(center_h**2 + center_w**2)
        band_width = max_radius / self.num_bands
        
        self.band_masks = []
        for i in range(self.num_bands):
            inner = i * band_width
            outer = (i + 1) * band_width
            mask = (self.distance_map >= inner) & (self.distance_map < outer)
            self.band_masks.append(mask)
    
    @log_execution_time
    def extract(self, image: Union[str, Path, np.ndarray]) -> FFTFeatures:
        """
        Extract FFT-based forensic features from an image.
        
        Parameters
        ----------
        image : str, Path, or np.ndarray
            Input image path or array
            
        Returns
        -------
        FFTFeatures
            Comprehensive FFT feature set
        """
        # Load image if path
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Could not read image: {image}")
        else:
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img = image.copy()
        
        # Standardize size
        img = cv2.resize(img, (self.image_size, self.image_size))
        img = img.astype(np.float64)
        
        # Apply window function to reduce edge artifacts
        window = self._create_window(img.shape)
        img_windowed = img * window
        
        # Compute FFT
        fft_result = fftshift(fft2(img_windowed))
        magnitude = np.abs(fft_result)
        log_magnitude = np.log(magnitude + 1e-10)
        
        # Extract features
        features = self._compute_features(magnitude, log_magnitude)
        
        return features
    
    def _create_window(self, shape: Tuple[int, int]) -> np.ndarray:
        """Create 2D Hanning window."""
        h, w = shape
        window_h = np.hanning(h)
        window_w = np.hanning(w)
        return np.outer(window_h, window_w)
    
    def _compute_features(
        self,
        magnitude: np.ndarray,
        log_magnitude: np.ndarray
    ) -> FFTFeatures:
        """Compute all FFT-based features."""
        
        # Basic energy features
        low_freq_energy = np.mean(log_magnitude[self.low_freq_mask])
        high_freq_energy = np.mean(log_magnitude[self.high_freq_mask])
        spectral_ratio = high_freq_energy / (low_freq_energy + 1e-10)
        
        # Spectral entropy
        norm_mag = magnitude / (np.sum(magnitude) + 1e-10)
        spectral_entropy = -np.sum(norm_mag * np.log(norm_mag + 1e-10))
        
        # Azimuthally averaged power spectrum (radial profile)
        azimuthal_profile = self._compute_radial_profile(magnitude)
        
        # Band energies
        band_energies = self._compute_band_energies(log_magnitude)
        
        # Spectral statistics
        spectral_flatness = self._compute_spectral_flatness(magnitude)
        spectral_centroid = self._compute_spectral_centroid(magnitude)
        spectral_rolloff = self._compute_spectral_rolloff(magnitude)
        
        # GAN fingerprint detection
        gan_fingerprint_score = self._detect_gan_fingerprint(magnitude)
        
        return FFTFeatures(
            low_freq_energy=low_freq_energy,
            high_freq_energy=high_freq_energy,
            spectral_ratio=spectral_ratio,
            spectral_entropy=spectral_entropy,
            azimuthal_profile=azimuthal_profile,
            band_energies=band_energies,
            spectral_flatness=spectral_flatness,
            spectral_centroid=spectral_centroid,
            spectral_rolloff=spectral_rolloff,
            gan_fingerprint_score=gan_fingerprint_score
        )
    
    def _compute_radial_profile(
        self,
        magnitude: np.ndarray,
        num_bins: int = 32
    ) -> np.ndarray:
        """
        Compute azimuthally averaged radial profile.
        
        This is the average magnitude at each distance from center,
        useful for detecting periodic GAN artifacts.
        """
        center = np.array(magnitude.shape) // 2
        max_radius = min(center)
        
        # Bin the distances
        bin_edges = np.linspace(0, max_radius, num_bins + 1)
        profile = np.zeros(num_bins)
        
        for i in range(num_bins):
            mask = (self.distance_map >= bin_edges[i]) & \
                   (self.distance_map < bin_edges[i + 1])
            if np.any(mask):
                profile[i] = np.mean(magnitude[mask])
        
        # Normalize
        profile = profile / (np.max(profile) + 1e-10)
        
        return profile
    
    def _compute_band_energies(self, log_magnitude: np.ndarray) -> np.ndarray:
        """Compute energy in each frequency band."""
        energies = np.zeros(self.num_bands)
        
        for i, mask in enumerate(self.band_masks):
            if np.any(mask):
                energies[i] = np.mean(log_magnitude[mask])
        
        return energies
    
    def _compute_spectral_flatness(self, magnitude: np.ndarray) -> float:
        """
        Compute spectral flatness (Wiener entropy).
        
        Measures how noise-like vs. tonal the spectrum is.
        AI-generated images often have different flatness patterns.
        """
        magnitude_flat = magnitude.flatten()
        magnitude_flat = magnitude_flat[magnitude_flat > 0]
        
        if len(magnitude_flat) == 0:
            return 0.0
        
        geometric_mean = np.exp(np.mean(np.log(magnitude_flat + 1e-10)))
        arithmetic_mean = np.mean(magnitude_flat)
        
        return geometric_mean / (arithmetic_mean + 1e-10)
    
    def _compute_spectral_centroid(self, magnitude: np.ndarray) -> float:
        """
        Compute spectral centroid (center of mass of spectrum).
        
        Indicates the "brightness" of the frequency content.
        """
        total = np.sum(magnitude)
        if total < 1e-10:
            return 0.0
        
        weighted_sum = np.sum(magnitude * self.distance_map)
        return weighted_sum / total
    
    def _compute_spectral_rolloff(
        self,
        magnitude: np.ndarray,
        percentile: float = 0.85
    ) -> float:
        """
        Compute spectral rolloff frequency.
        
        The frequency below which a certain percentage of energy is contained.
        """
        total_energy = np.sum(magnitude)
        threshold = percentile * total_energy
        
        # Sort distances and corresponding magnitudes
        flat_distances = self.distance_map.flatten()
        flat_magnitude = magnitude.flatten()
        
        sort_idx = np.argsort(flat_distances)
        sorted_distances = flat_distances[sort_idx]
        cumsum = np.cumsum(flat_magnitude[sort_idx])
        
        # Find rolloff point
        rolloff_idx = np.searchsorted(cumsum, threshold)
        if rolloff_idx >= len(sorted_distances):
            rolloff_idx = len(sorted_distances) - 1
        
        return sorted_distances[rolloff_idx]
    
    def _detect_gan_fingerprint(self, magnitude: np.ndarray) -> float:
        """
        Detect GAN-specific frequency fingerprints.
        
        Many GANs produce characteristic periodic patterns in the
        frequency domain due to upsampling operations.
        
        Returns
        -------
        float
            Score indicating likelihood of GAN fingerprint (0-1)
        """
        center = np.array(magnitude.shape) // 2
        
        # Check for periodic peaks (GAN artifacts from upsampling)
        # Look for peaks at regular intervals from center
        
        # Analyze horizontal and vertical slices through center
        h_slice = magnitude[center[0], :]
        v_slice = magnitude[:, center[1]]
        
        # Compute autocorrelation to detect periodicity
        h_autocorr = np.correlate(h_slice, h_slice, mode='full')
        v_autocorr = np.correlate(v_slice, v_slice, mode='full')
        
        # Normalize
        h_autocorr = h_autocorr / (np.max(h_autocorr) + 1e-10)
        v_autocorr = v_autocorr / (np.max(v_autocorr) + 1e-10)
        
        # Find secondary peaks (indicating periodicity)
        # Skip the central peak
        mid = len(h_autocorr) // 2
        h_secondary = np.max(h_autocorr[mid + 10:])
        v_secondary = np.max(v_autocorr[mid + 10:])
        
        # Higher secondary peaks suggest periodic structure (GAN fingerprint)
        fingerprint_score = (h_secondary + v_secondary) / 2
        
        return float(np.clip(fingerprint_score, 0, 1))
    
    def visualize_spectrum(
        self,
        image: Union[str, Path, np.ndarray],
        save_path: Optional[Union[str, Path]] = None
    ) -> np.ndarray:
        """
        Create visualization of the FFT spectrum.
        
        Parameters
        ----------
        image : str, Path, or np.ndarray
            Input image
        save_path : str or Path, optional
            Path to save visualization
            
        Returns
        -------
        np.ndarray
            Visualization image
        """
        import matplotlib.pyplot as plt
        
        # Load and process image
        if isinstance(image, (str, Path)):
            img = cv2.imread(str(image), cv2.IMREAD_GRAYSCALE)
        else:
            if len(image.shape) == 3:
                img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                img = image.copy()
        
        img = cv2.resize(img, (self.image_size, self.image_size))
        
        # Compute FFT
        fft_result = fftshift(fft2(img))
        magnitude = np.log(np.abs(fft_result) + 1e-10)
        
        # Create visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # FFT magnitude
        axes[1].imshow(magnitude, cmap='viridis')
        axes[1].set_title('FFT Magnitude Spectrum')
        axes[1].axis('off')
        
        # Radial profile
        features = self.extract(img)
        axes[2].plot(features.azimuthal_profile)
        axes[2].set_title('Radial Power Profile')
        axes[2].set_xlabel('Frequency (bins)')
        axes[2].set_ylabel('Normalized Power')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved spectrum visualization to {save_path}")
        
        # Convert to numpy array
        fig.canvas.draw()
        vis_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        vis_array = vis_array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        plt.close(fig)
        
        return vis_array


def extract_fft_features(image_path: Union[str, Path]) -> np.ndarray:
    """
    Convenience function to extract FFT features from an image.
    
    Parameters
    ----------
    image_path : str or Path
        Path to input image
        
    Returns
    -------
    np.ndarray
        Feature vector
    """
    extractor = FFTFeatureExtractor()
    features = extractor.extract(image_path)
    return features.to_array()
