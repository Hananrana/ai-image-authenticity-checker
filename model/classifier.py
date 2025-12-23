"""
Enhanced Classifier Module
==========================

Industry-level ML classifiers with cross-validation and hyperparameter tuning.
"""

import os
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List, Union
from dataclasses import dataclass
import joblib

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

try:
    from lightgbm import LGBMClassifier
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False

try:
    from config import MODEL_CONFIG, MODEL_DIR, LABEL_NAMES
except ImportError:
    MODEL_CONFIG = None
    MODEL_DIR = Path("saved_models")
    LABEL_NAMES = {0: "Real", 1: "AI-Generated"}

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time
from features.feature_fusion import FeatureFusion

logger = get_logger(__name__)


@dataclass
class TrainingResults:
    """Container for training results."""
    accuracy: float
    auc_score: float
    cv_scores: np.ndarray
    classification_report: str
    confusion_matrix: np.ndarray
    feature_importance: Optional[np.ndarray] = None


class AIImageClassifier:
    """
    Multi-algorithm classifier for AI image detection.
    
    Supports SVM, Random Forest, XGBoost, and LightGBM with
    automatic hyperparameter tuning and cross-validation.
    """
    
    SUPPORTED_ALGORITHMS = ["svm", "rf", "xgboost", "lightgbm"]
    
    def __init__(
        self,
        algorithm: str = "rf",
        random_state: int = 42,
        **kwargs
    ):
        if MODEL_CONFIG:
            self.random_state = MODEL_CONFIG.random_seed
        else:
            self.random_state = random_state
        
        self.algorithm = algorithm.lower()
        self.model = self._create_model(self.algorithm, **kwargs)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_importance_ = None
        
        logger.info(f"Initialized {algorithm.upper()} classifier")
    
    def _create_model(self, algorithm: str, **kwargs) -> Any:
        """Create model instance."""
        if algorithm == "svm":
            return SVC(
                kernel=kwargs.get("kernel", "rbf"),
                C=kwargs.get("C", 1.0),
                probability=True,
                random_state=self.random_state
            )
        elif algorithm == "rf":
            return RandomForestClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", None),
                random_state=self.random_state,
                n_jobs=-1
            )
        elif algorithm == "xgboost":
            if not XGB_AVAILABLE:
                raise ImportError("XGBoost not installed")
            return XGBClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                max_depth=kwargs.get("max_depth", 6),
                random_state=self.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            )
        elif algorithm == "lightgbm":
            if not LGBM_AVAILABLE:
                raise ImportError("LightGBM not installed")
            return LGBMClassifier(
                n_estimators=kwargs.get("n_estimators", 200),
                num_leaves=kwargs.get("num_leaves", 31),
                random_state=self.random_state,
                verbose=-1
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")
    
    @log_execution_time
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        validation_split: float = 0.2
    ) -> TrainingResults:
        """
        Train the classifier with cross-validation.
        
        Parameters
        ----------
        X : np.ndarray
            Feature matrix (n_samples, n_features)
        y : np.ndarray
            Labels (n_samples,)
        validation_split : float
            Fraction for validation
            
        Returns
        -------
        TrainingResults
            Training metrics and results
        """
        logger.info(f"Training {self.algorithm} on {len(X)} samples")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, stratify=y, random_state=self.random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Cross-validation
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        cv_scores = cross_val_score(self.model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")
        
        # Train on full training set
        self.model.fit(X_train_scaled, y_train)
        self.is_fitted = True
        
        # Evaluate on validation
        y_pred = self.model.predict(X_val_scaled)
        y_prob = self.model.predict_proba(X_val_scaled)[:, 1]
        
        accuracy = accuracy_score(y_val, y_pred)
        auc = roc_auc_score(y_val, y_prob)
        report = classification_report(y_val, y_pred, target_names=list(LABEL_NAMES.values()))
        cm = confusion_matrix(y_val, y_pred)
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance_ = self.model.feature_importances_
        
        logger.info(f"Validation Accuracy: {accuracy:.4f}, AUC: {auc:.4f}")
        
        return TrainingResults(
            accuracy=accuracy,
            auc_score=auc,
            cv_scores=cv_scores,
            classification_report=report,
            confusion_matrix=cm,
            feature_importance=self.feature_importance_
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self.is_fitted:
            raise RuntimeError("Model not fitted")
        X_scaled = self.scaler.transform(X.reshape(1, -1) if X.ndim == 1 else X)
        return self.model.predict_proba(X_scaled)
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'algorithm': self.algorithm,
            'is_fitted': self.is_fitted,
            'feature_importance': self.feature_importance_
        }
        joblib.dump(model_data, path)
        logger.info(f"Saved model to {path}")
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'AIImageClassifier':
        """Load model from disk."""
        model_data = joblib.load(path)
        instance = cls(algorithm=model_data['algorithm'])
        instance.model = model_data['model']
        instance.scaler = model_data['scaler']
        instance.is_fitted = model_data['is_fitted']
        instance.feature_importance_ = model_data.get('feature_importance')
        logger.info(f"Loaded model from {path}")
        return instance


def train_classifier(
    real_dir: str,
    fake_dir: str,
    algorithm: str = "rf",
    save_path: Optional[str] = None
) -> Tuple[AIImageClassifier, TrainingResults]:
    """
    Train a classifier on real and fake image directories.
    
    Parameters
    ----------
    real_dir : str
        Directory with real images
    fake_dir : str
        Directory with AI-generated images
    algorithm : str
        Algorithm to use
    save_path : str, optional
        Path to save trained model
        
    Returns
    -------
    Tuple[AIImageClassifier, TrainingResults]
        Trained model and results
    """
    from features.feature_fusion import FeatureFusion
    from tqdm import tqdm
    
    fusion = FeatureFusion(include_deep=False)
    X, y = [], []
    
    # Load real images (label 0)
    real_path = Path(real_dir)
    fake_path = Path(fake_dir)
    
    for img_path in tqdm(list(real_path.glob("*")), desc="Processing real images"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            try:
                features = fusion.extract(img_path).to_array(False)
                X.append(features)
                y.append(0)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
    
    # Load fake images (label 1)
    for img_path in tqdm(list(fake_path.glob("*")), desc="Processing fake images"):
        if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
            try:
                features = fusion.extract(img_path).to_array(False)
                X.append(features)
                y.append(1)
            except Exception as e:
                logger.warning(f"Failed to process {img_path}: {e}")
    
    X = np.array(X)
    y = np.array(y)
    
    logger.info(f"Loaded {len(X)} images ({sum(y==0)} real, {sum(y==1)} fake)")
    
    # Train
    classifier = AIImageClassifier(algorithm=algorithm)
    results = classifier.fit(X, y)
    
    # Save
    if save_path:
        classifier.save(save_path)
    
    return classifier, results


def load_classifier(path: Union[str, Path]) -> AIImageClassifier:
    """Load a trained classifier."""
    return AIImageClassifier.load(path)
