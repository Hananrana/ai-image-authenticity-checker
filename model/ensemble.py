"""
Ensemble Classifier
===================

Meta-ensemble combining multiple ML algorithms for robust predictions.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Any, Union
import joblib

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

try:
    from config import MODEL_CONFIG, MODEL_DIR
except ImportError:
    MODEL_CONFIG = None
    MODEL_DIR = Path("saved_models")

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger

logger = get_logger(__name__)


class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Weighted ensemble of multiple classifiers.
    
    Combines SVM, Random Forest, XGBoost, and LightGBM
    using weighted voting for robust predictions.
    """
    
    def __init__(
        self,
        weights: Optional[List[float]] = None,
        include_algorithms: Optional[List[str]] = None
    ):
        self.include_algorithms = include_algorithms or ["svm", "rf", "xgboost", "lightgbm"]
        
        if MODEL_CONFIG:
            self.weights = weights or MODEL_CONFIG.ensemble_weights[:len(self.include_algorithms)]
        else:
            self.weights = weights or [1.0 / len(self.include_algorithms)] * len(self.include_algorithms)
        
        # Normalize weights
        self.weights = np.array(self.weights)
        self.weights = self.weights / self.weights.sum()
        
        self.classifiers = {}
        self.scaler = StandardScaler()
        self.is_fitted = False
        
        logger.info(f"Initialized ensemble with {self.include_algorithms}")
    
    def _create_classifiers(self) -> Dict[str, Any]:
        """Create all classifier instances."""
        from .classifier import AIImageClassifier
        
        classifiers = {}
        for algo in self.include_algorithms:
            try:
                classifiers[algo] = AIImageClassifier(algorithm=algo)
                logger.debug(f"Created {algo} classifier")
            except ImportError as e:
                logger.warning(f"Could not create {algo}: {e}")
        
        return classifiers
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EnsembleClassifier':
        """Train all classifiers in the ensemble."""
        logger.info(f"Training ensemble on {len(X)} samples")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and train classifiers
        self.classifiers = {}
        for algo in self.include_algorithms:
            try:
                from .classifier import AIImageClassifier
                clf = AIImageClassifier(algorithm=algo)
                clf.fit(X, y)
                self.classifiers[algo] = clf
                logger.info(f"Trained {algo}")
            except Exception as e:
                logger.warning(f"Failed to train {algo}: {e}")
        
        # Adjust weights based on available classifiers
        available = list(self.classifiers.keys())
        if len(available) < len(self.include_algorithms):
            self.weights = np.array([1.0 / len(available)] * len(available))
        
        self.is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict using weighted voting."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")
        
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using weighted average."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")
        
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        # Collect predictions
        all_probs = []
        for i, (algo, clf) in enumerate(self.classifiers.items()):
            try:
                prob = clf.predict_proba(X)
                all_probs.append(prob * self.weights[i])
            except Exception as e:
                logger.warning(f"Prediction failed for {algo}: {e}")
        
        if not all_probs:
            raise RuntimeError("All classifiers failed")
        
        # Weighted average
        ensemble_prob = np.sum(all_probs, axis=0)
        
        return ensemble_prob
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Get predictions from each classifier."""
        predictions = {}
        for algo, clf in self.classifiers.items():
            try:
                predictions[algo] = clf.predict_proba(X)
            except Exception:
                continue
        return predictions
    
    def save(self, path: Union[str, Path]) -> Path:
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'classifiers': {k: v for k, v in self.classifiers.items()},
            'scaler': self.scaler,
            'weights': self.weights,
            'include_algorithms': self.include_algorithms,
            'is_fitted': self.is_fitted
        }
        joblib.dump(data, path)
        logger.info(f"Saved ensemble to {path}")
        return path
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> 'EnsembleClassifier':
        """Load ensemble from disk."""
        data = joblib.load(path)
        
        instance = cls(
            weights=list(data['weights']),
            include_algorithms=data['include_algorithms']
        )
        instance.classifiers = data['classifiers']
        instance.scaler = data['scaler']
        instance.is_fitted = data['is_fitted']
        
        logger.info(f"Loaded ensemble from {path}")
        return instance
