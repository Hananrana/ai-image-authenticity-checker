"""
Model Training Pipeline
=======================

Complete training pipeline with data loading, augmentation, and evaluation.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Union
from datetime import datetime
import json

from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from config import (MODEL_DIR, DATA_DIR, TRAIN_REAL_DIR, TRAIN_FAKE_DIR,
                       VALID_REAL_DIR, VALID_FAKE_DIR, TEST_REAL_DIR, TEST_FAKE_DIR,
                       MODEL_CONFIG, RESULTS_DIR, IMAGE_CONFIG, DL_CONFIG,
                       REAL_IMAGES_DIR, FAKE_IMAGES_DIR)
except ImportError:
    MODEL_DIR = Path("saved_models")
    DATA_DIR = Path("data")
    TRAIN_REAL_DIR = DATA_DIR / "train" / "real"
    TRAIN_FAKE_DIR = DATA_DIR / "train" / "fake"
    VALID_REAL_DIR = DATA_DIR / "valid" / "real"
    VALID_FAKE_DIR = DATA_DIR / "valid" / "fake"
    TEST_REAL_DIR = DATA_DIR / "test" / "real"
    TEST_FAKE_DIR = DATA_DIR / "test" / "fake"
    # Legacy paths
    REAL_IMAGES_DIR = TRAIN_REAL_DIR
    FAKE_IMAGES_DIR = TRAIN_FAKE_DIR
    RESULTS_DIR = Path("outputs/results")
    MODEL_CONFIG = None
    IMAGE_CONFIG = None
    DL_CONFIG = None

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


class ModelTrainer:
    """
    Complete training pipeline for AI image detection models.
    
    Supports both classical ML (SVM, Random Forest, XGBoost) and 
    deep learning (via train_gpu.py) training modes.
    
    For deep learning training with GPU acceleration, use:
        python train_gpu.py --epochs 50
    """
    
    def __init__(
        self,
        real_dir: Optional[Path] = None,
        fake_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        include_deep: bool = False,
        use_kaggle_structure: bool = True
    ):
        # Use Kaggle structure (train/real, train/fake) by default
        if use_kaggle_structure:
            self.real_dir = Path(real_dir) if real_dir else TRAIN_REAL_DIR
            self.fake_dir = Path(fake_dir) if fake_dir else TRAIN_FAKE_DIR
            self.val_real_dir = VALID_REAL_DIR
            self.val_fake_dir = VALID_FAKE_DIR
            self.test_real_dir = TEST_REAL_DIR
            self.test_fake_dir = TEST_FAKE_DIR
        else:
            # Legacy flat structure
            self.real_dir = Path(real_dir) if real_dir else REAL_IMAGES_DIR
            self.fake_dir = Path(fake_dir) if fake_dir else FAKE_IMAGES_DIR
            self.val_real_dir = None
            self.val_fake_dir = None
            self.test_real_dir = None
            self.test_fake_dir = None
        
        self.output_dir = Path(output_dir) if output_dir else MODEL_DIR
        self.include_deep = include_deep
        self.use_kaggle_structure = use_kaggle_structure
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.feature_extractor = None
        
        logger.info(f"Initialized trainer with real={self.real_dir}, fake={self.fake_dir}")
        if use_kaggle_structure:
            logger.info("Using Kaggle dataset structure (train/valid/test splits)")
    
    def load_dataset(self, max_samples: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """Load and extract features from dataset."""
        from features.feature_fusion import FeatureFusion
        
        self.feature_extractor = FeatureFusion(include_deep=self.include_deep)
        
        X, y = [], []
        supported = IMAGE_CONFIG.supported_formats if IMAGE_CONFIG else \
            ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        
        # Load real images
        real_files = [f for f in self.real_dir.iterdir() if f.suffix.lower() in supported]
        if max_samples:
            real_files = real_files[:max_samples // 2]
        
        for img_path in tqdm(real_files, desc="Loading real images"):
            try:
                features = self.feature_extractor.extract(img_path).to_array(self.include_deep)
                X.append(features)
                y.append(0)
            except Exception as e:
                logger.debug(f"Skipping {img_path}: {e}")
        
        # Load fake images
        fake_files = [f for f in self.fake_dir.iterdir() if f.suffix.lower() in supported]
        if max_samples:
            fake_files = fake_files[:max_samples // 2]
        
        for img_path in tqdm(fake_files, desc="Loading fake images"):
            try:
                features = self.feature_extractor.extract(img_path).to_array(self.include_deep)
                X.append(features)
                y.append(1)
            except Exception as e:
                logger.debug(f"Skipping {img_path}: {e}")
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Loaded {len(X)} samples ({np.sum(y==0)} real, {np.sum(y==1)} fake)")
        
        return X, y
    
    def split_data(
        self,
        X: np.ndarray,
        y: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15
    ):
        """Split data into train/val/test sets."""
        test_ratio = 1.0 - train_ratio - val_ratio
        
        # First split: train+val vs test
        X_trainval, self.X_test, y_trainval, self.y_test = train_test_split(
            X, y, test_size=test_ratio, stratify=y, random_state=42
        )
        
        # Second split: train vs val
        val_adjusted = val_ratio / (train_ratio + val_ratio)
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            X_trainval, y_trainval, test_size=val_adjusted, stratify=y_trainval, random_state=42
        )
        
        logger.info(f"Split: train={len(self.X_train)}, val={len(self.X_val)}, test={len(self.X_test)}")
    
    @log_execution_time
    def train(
        self,
        algorithm: str = "rf",
        use_ensemble: bool = False
    ) -> Dict:
        """
        Train the model.
        
        Parameters
        ----------
        algorithm : str
            Algorithm for single classifier
        use_ensemble : bool
            Use ensemble instead of single classifier
            
        Returns
        -------
        Dict
            Training results and metrics
        """
        if self.X_train is None:
            raise RuntimeError("Call load_dataset and split_data first")
        
        if use_ensemble:
            from .ensemble import EnsembleClassifier
            model = EnsembleClassifier()
            model.fit(self.X_train, self.y_train)
        else:
            from .classifier import AIImageClassifier
            model = AIImageClassifier(algorithm=algorithm)
            model.fit(self.X_train, self.y_train)
        
        # Evaluate on test set
        y_pred = model.predict(self.X_test)
        y_prob = model.predict_proba(self.X_test)[:, 1]
        
        from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
        
        results = {
            'algorithm': 'ensemble' if use_ensemble else algorithm,
            'test_accuracy': float(accuracy_score(self.y_test, y_pred)),
            'test_auc': float(roc_auc_score(self.y_test, y_prob)),
            'train_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"Test Accuracy: {results['test_accuracy']:.4f}")
        logger.info(f"Test AUC: {results['test_auc']:.4f}")
        
        # Save model
        model_name = f"model_{algorithm}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        model_path = self.output_dir / f"{model_name}.joblib"
        model.save(model_path)
        results['model_path'] = str(model_path)
        
        # Save results
        results_path = RESULTS_DIR / f"{model_name}_results.json"
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        return results
    
    def run_full_pipeline(
        self,
        max_samples: Optional[int] = None,
        algorithm: str = "rf"
    ) -> Dict:
        """Run complete training pipeline."""
        logger.info("Starting full training pipeline")
        
        # Load data
        X, y = self.load_dataset(max_samples)
        
        # Split
        self.split_data(X, y)
        
        # Train
        results = self.train(algorithm=algorithm)
        
        logger.info("Training pipeline completed")
        return results
