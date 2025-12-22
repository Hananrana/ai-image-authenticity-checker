import cv2
import numpy as np
from scipy.fft import fft2, fftshift


def extract_fft_features(image_path: str) -> np.ndarray:
    """
    Extract frequency-domain forensic features from an image.

    Parameters
    ----------
    image_path : str
        Path to input image.

    Returns
    -------
    np.ndarray
        Feature vector containing:
        [low_freq_energy, high_freq_energy, spectral_ratio, spectral_entropy]
    """
    # Load image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    # Standardize size for frequency consistency
    img = cv2.resize(img, (256, 256))

    # FFT and magnitude spectrum
    fft = fftshift(fft2(img))
    magnitude = np.log(np.abs(fft) + 1e-8)

    h, w = magnitude.shape
    center_h, center_w = h // 2, w // 2

    # Low-frequency (central region)
    low_freq = magnitude[
        center_h - 32:center_h + 32,
        center_w - 32:center_w + 32
    ]

    # High-frequency (outer regions)
    high_freq = magnitude.copy()
    high_freq[
        center_h - 32:center_h + 32,
        center_w - 32:center_w + 32
    ] = 0

    low_freq_energy = np.mean(low_freq)
    high_freq_energy = np.mean(high_freq)

    spectral_ratio = high_freq_energy / (low_freq_energy + 1e-8)

    # Spectral entropy (distribution irregularity)
    norm_mag = magnitude / np.sum(magnitude)
    spectral_entropy = -np.sum(norm_mag * np.log(norm_mag + 1e-8))

    return np.array([
        low_freq_energy,
        high_freq_energy,
        spectral_ratio,
        spectral_entropy
    ])
