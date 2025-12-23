"""
AI Image Authenticity Checker - Configuration
=============================================

Central configuration file for all project settings, paths, and hyperparameters.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories - Kaggle 140k Real and Fake Faces structure
DATA_DIR = PROJECT_ROOT / "data"

# Training data (train/real, train/fake)
TRAIN_DIR = DATA_DIR / "train"
TRAIN_REAL_DIR = TRAIN_DIR / "real"
TRAIN_FAKE_DIR = TRAIN_DIR / "fake"

# Validation data (valid/real, valid/fake)
VALID_DIR = DATA_DIR / "valid"
VALID_REAL_DIR = VALID_DIR / "real"
VALID_FAKE_DIR = VALID_DIR / "fake"

# Test data (test/real, test/fake)
TEST_DIR = DATA_DIR / "test"
TEST_REAL_DIR = TEST_DIR / "real"
TEST_FAKE_DIR = TEST_DIR / "fake"

# Legacy paths for backward compatibility
REAL_IMAGES_DIR = TRAIN_REAL_DIR
FAKE_IMAGES_DIR = TRAIN_FAKE_DIR
PROCESSED_DIR = DATA_DIR / "processed"

# Model directories
MODEL_DIR = PROJECT_ROOT / "saved_models"
CHECKPOINT_DIR = MODEL_DIR / "checkpoints"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOGS_DIR = OUTPUT_DIR / "logs"
TENSORBOARD_DIR = LOGS_DIR / "tensorboard"
RESULTS_DIR = OUTPUT_DIR / "results"
VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"

# Create directories if they don't exist
for directory in [TRAIN_REAL_DIR, TRAIN_FAKE_DIR, VALID_REAL_DIR, VALID_FAKE_DIR,
                  TEST_REAL_DIR, TEST_FAKE_DIR, PROCESSED_DIR, MODEL_DIR, 
                  CHECKPOINT_DIR, OUTPUT_DIR, LOGS_DIR, TENSORBOARD_DIR,
                  RESULTS_DIR, VISUALIZATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# IMAGE CONFIGURATION
# =============================================================================

@dataclass
class ImageConfig:
    """Image processing configuration."""
    
    # Standard image size for processing
    target_size: Tuple[int, int] = (256, 256)
    
    # Deep learning model input size
    model_input_size: Tuple[int, int] = (224, 224)
    
    # Supported image formats
    supported_formats: Tuple[str, ...] = ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tiff')
    
    # JPEG quality for ELA analysis
    ela_quality: int = 90
    
    # Maximum file size (MB)
    max_file_size_mb: float = 50.0


IMAGE_CONFIG = ImageConfig()


# =============================================================================
# FEATURE EXTRACTION CONFIGURATION
# =============================================================================

@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    
    # FFT Features
    fft_image_size: int = 256
    fft_low_freq_radius: int = 32
    fft_num_bands: int = 8
    
    # ELA Features
    ela_quality_levels: List[int] = field(default_factory=lambda: [90, 75, 50])
    ela_block_size: int = 8
    
    # Texture Features (LBP)
    lbp_radius: int = 3
    lbp_n_points: int = 24
    
    # GLCM Features
    glcm_distances: List[int] = field(default_factory=lambda: [1, 2, 5])
    glcm_angles: List[float] = field(default_factory=lambda: [0, 0.785, 1.571, 2.356])
    
    # Noise Features
    noise_patch_size: int = 64
    noise_num_patches: int = 16
    
    # Deep Features
    deep_model_name: str = "efficientnet_b0"
    deep_feature_layer: str = "avgpool"


FEATURE_CONFIG = FeatureConfig()


# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

@dataclass
class ModelConfig:
    """Machine learning model configuration."""
    
    # Random seed for reproducibility
    random_seed: int = 42
    
    # Train/Val/Test split ratios
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    
    # Cross-validation
    cv_folds: int = 5
    
    # Classical ML Models
    svm_kernel: str = "rbf"
    svm_C: float = 1.0
    rf_n_estimators: int = 200
    rf_max_depth: Optional[int] = None
    xgb_n_estimators: int = 200
    xgb_max_depth: int = 6
    lgb_n_estimators: int = 200
    lgb_num_leaves: int = 31
    
    # Ensemble weights (SVM, RF, XGB, LGB, Deep)
    ensemble_weights: List[float] = field(default_factory=lambda: [0.2, 0.2, 0.25, 0.15, 0.2])


MODEL_CONFIG = ModelConfig()


# =============================================================================
# DEEP LEARNING CONFIGURATION
# =============================================================================

@dataclass
class DeepLearningConfig:
    """
    Deep learning training configuration.
    Optimized for NVIDIA RTX 3060 (12GB VRAM, 3584 CUDA cores).
    """
    
    # Model architecture
    backbone: str = "efficientnet_b4"  # Good balance of accuracy and VRAM usage
    pretrained: bool = True
    num_classes: int = 2
    dropout_rate: float = 0.3
    
    # Training - Optimized for RTX 3060 (12GB VRAM)
    batch_size: int = 24  # Optimal for 12GB VRAM with EfficientNet-B4
    num_epochs: int = 100  # More epochs with early stopping
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    
    # Gradient accumulation for larger effective batch size
    gradient_accumulation_steps: int = 2  # Effective batch = 48
    
    # Learning rate scheduler
    scheduler: str = "cosine"  # "cosine", "step", "plateau"
    warmup_epochs: int = 5
    
    # Early stopping
    early_stopping_patience: int = 15
    min_delta: float = 1e-4
    
    # Mixed precision training - Leverages RTX 3060 Tensor Cores
    use_amp: bool = True  # ~2x speedup with mixed precision
    
    # Data augmentation
    augmentation_prob: float = 0.5
    
    # DataLoader settings - Optimized for 6-core CPU
    num_workers: int = 4  # Optimal for Ryzen 5 5600X
    pin_memory: bool = True  # Faster GPU data transfer
    prefetch_factor: int = 2  # Prefetch batches
    
    # Device configuration
    device: str = "cuda"  # Fallback to CPU if CUDA unavailable
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n_models: int = 3


DL_CONFIG = DeepLearningConfig()


# =============================================================================
# DATASET CONFIGURATION
# =============================================================================

@dataclass
class DatasetConfig:
    """Dataset download and preparation configuration."""
    
    # Sample sizes for quick testing
    quick_test_samples: int = 100
    
    # Full dataset sizes
    real_images_count: int = 10000
    fake_images_count: int = 10000
    
    # Dataset sources
    coco_url: str = "http://images.cocodataset.org/zips/unlabeled2017.zip"
    
    # Hugging Face datasets for AI-generated images
    hf_stable_diffusion: str = "poloclub/diffusiondb"
    hf_midjourney: str = "wanng/midjourney-v5-202304-clean"
    
    # Data split
    stratify: bool = True
    shuffle: bool = True


DATASET_CONFIG = DatasetConfig()


# =============================================================================
# API CONFIGURATION
# =============================================================================

@dataclass
class APIConfig:
    """API server configuration."""
    
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Rate limiting
    rate_limit_requests: int = 100
    rate_limit_period: int = 60  # seconds
    
    # File upload
    max_upload_size_mb: int = 50
    
    # Response
    include_explanations: bool = True
    include_confidence: bool = True


API_CONFIG = APIConfig()


# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

@dataclass
class LoggingConfig:
    """Logging configuration."""
    
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"
    
    # File logging
    log_to_file: bool = True
    log_file: Path = LOGS_DIR / "app.log"
    max_log_size_mb: int = 10
    backup_count: int = 5


LOGGING_CONFIG = LoggingConfig()


# =============================================================================
# LABELS AND CONSTANTS
# =============================================================================

# Class labels
LABEL_REAL = 0
LABEL_FAKE = 1

LABEL_NAMES = {
    LABEL_REAL: "Real",
    LABEL_FAKE: "AI-Generated"
}

# Confidence thresholds
CONFIDENCE_THRESHOLD_LOW = 0.3
CONFIDENCE_THRESHOLD_HIGH = 0.7

# Model versioning
MODEL_VERSION = "1.0.0"
