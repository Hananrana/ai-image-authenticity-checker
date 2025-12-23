"""
Image Prediction Module
=======================

Complete inference pipeline with confidence scoring and explainability.
"""

import numpy as np
from pathlib import Path
from typing import Union, Optional, Dict, List, Tuple
from dataclasses import dataclass
import json

try:
    from config import MODEL_DIR, LABEL_NAMES, CONFIDENCE_THRESHOLD_LOW, CONFIDENCE_THRESHOLD_HIGH
except ImportError:
    MODEL_DIR = Path("saved_models")
    LABEL_NAMES = {0: "Real", 1: "AI-Generated"}
    CONFIDENCE_THRESHOLD_LOW = 0.3
    CONFIDENCE_THRESHOLD_HIGH = 0.7

import sys
sys.path.append(str(Path(__file__).parent.parent))
from utils.logger import get_logger, log_execution_time

logger = get_logger(__name__)


@dataclass
class PredictionResult:
    """Container for prediction results."""
    image_path: str
    prediction: str  # "Real" or "AI-Generated"
    label: int  # 0 or 1
    confidence: float  # 0-1
    confidence_level: str  # "low", "medium", "high"
    probabilities: Dict[str, float]
    feature_contributions: Optional[Dict[str, float]] = None
    
    def to_dict(self) -> Dict:
        return {
            'image_path': self.image_path,
            'prediction': self.prediction,
            'label': self.label,
            'confidence': round(self.confidence, 4),
            'confidence_level': self.confidence_level,
            'probabilities': {k: round(v, 4) for k, v in self.probabilities.items()},
            'feature_contributions': self.feature_contributions
        }


class ImagePredictor:
    """
    Production-ready image prediction system.
    
    Provides predictions with confidence scoring, calibration,
    and optional explainability.
    """
    
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        include_deep: bool = False
    ):
        self.include_deep = include_deep
        self.model = None
        self.feature_extractor = None
        
        if model_path:
            self.load_model(model_path)
        
        self._init_feature_extractor()
    
    def _init_feature_extractor(self):
        """Initialize feature extraction pipeline."""
        from features.feature_fusion import FeatureFusion
        self.feature_extractor = FeatureFusion(include_deep=self.include_deep)
        logger.debug("Initialized feature extractor")
    
    def load_model(self, model_path: Union[str, Path]):
        """Load trained model."""
        from model.classifier import AIImageClassifier
        from model.ensemble import EnsembleClassifier
        import joblib
        
        model_path = Path(model_path)
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        try:
            # Try loading as AIImageClassifier
            self.model = AIImageClassifier.load(model_path)
        except Exception:
            try:
                # Try loading as EnsembleClassifier
                self.model = EnsembleClassifier.load(model_path)
            except Exception:
                # Try loading as raw model
                self.model = joblib.load(model_path)
        
        logger.info(f"Loaded model from {model_path}")
    
    @log_execution_time
    def predict(
        self,
        image: Union[str, Path, np.ndarray],
        return_features: bool = False
    ) -> PredictionResult:
        """
        Predict whether an image is real or AI-generated.
        
        Parameters
        ----------
        image : str, Path, or np.ndarray
            Input image
        return_features : bool
            Include feature contributions in result
            
        Returns
        -------
        PredictionResult
            Prediction with confidence and metadata
        """
        if self.model is None:
            raise RuntimeError("No model loaded. Call load_model() first.")
        
        # Get image path string
        image_path = str(image) if isinstance(image, (str, Path)) else "array_input"
        
        # Extract features
        features = self.feature_extractor.extract(image).to_array(self.include_deep)
        
        # Predict
        proba = self.model.predict_proba(features.reshape(1, -1))[0]
        label = int(proba[1] >= 0.5)
        confidence = proba[label]
        
        # Determine confidence level
        if confidence < CONFIDENCE_THRESHOLD_LOW + 0.2:
            confidence_level = "low"
        elif confidence < CONFIDENCE_THRESHOLD_HIGH:
            confidence_level = "medium"
        else:
            confidence_level = "high"
        
        # Feature contributions (if supported)
        feature_contributions = None
        if return_features and hasattr(self.model, 'feature_importance_'):
            if self.model.feature_importance_ is not None:
                names = self.feature_extractor.get_feature_names()
                importance = self.model.feature_importance_
                top_k = 10
                top_indices = np.argsort(importance)[-top_k:][::-1]
                feature_contributions = {
                    names[i] if i < len(names) else f"feature_{i}": float(importance[i])
                    for i in top_indices
                }
        
        return PredictionResult(
            image_path=image_path,
            prediction=LABEL_NAMES[label],
            label=label,
            confidence=float(confidence),
            confidence_level=confidence_level,
            probabilities={
                LABEL_NAMES[0]: float(proba[0]),
                LABEL_NAMES[1]: float(proba[1])
            },
            feature_contributions=feature_contributions
        )
    
    def predict_batch(
        self,
        images: List[Union[str, Path]],
        show_progress: bool = True
    ) -> List[PredictionResult]:
        """
        Predict multiple images.
        
        Parameters
        ----------
        images : List[str or Path]
            List of image paths
        show_progress : bool
            Show progress bar
            
        Returns
        -------
        List[PredictionResult]
            Predictions for all images
        """
        from tqdm import tqdm
        
        results = []
        iterator = tqdm(images, desc="Predicting") if show_progress else images
        
        for img_path in iterator:
            try:
                result = self.predict(img_path)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to predict {img_path}: {e}")
                results.append(PredictionResult(
                    image_path=str(img_path),
                    prediction="Error",
                    label=-1,
                    confidence=0.0,
                    confidence_level="error",
                    probabilities={"Real": 0.0, "AI-Generated": 0.0}
                ))
        
        return results
    
    def export_results(
        self,
        results: List[PredictionResult],
        output_path: Union[str, Path],
        format: str = "json"
    ):
        """Export prediction results."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "json":
            data = [r.to_dict() for r in results]
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2)
        elif format == "csv":
            import csv
            with open(output_path, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=[
                    'image_path', 'prediction', 'label', 'confidence', 'confidence_level'
                ])
                writer.writeheader()
                for r in results:
                    writer.writerow({
                        'image_path': r.image_path,
                        'prediction': r.prediction,
                        'label': r.label,
                        'confidence': r.confidence,
                        'confidence_level': r.confidence_level
                    })
        
        logger.info(f"Exported {len(results)} results to {output_path}")


def predict_image(
    image_path: Union[str, Path],
    model_path: Optional[Union[str, Path]] = None
) -> PredictionResult:
    """
    Convenience function for single image prediction.
    
    Parameters
    ----------
    image_path : str or Path
        Path to image
    model_path : str or Path, optional
        Path to trained model
        
    Returns
    -------
    PredictionResult
        Prediction result
    """
    predictor = ImagePredictor(model_path=model_path)
    return predictor.predict(image_path)


def predict_batch(
    image_paths: List[Union[str, Path]],
    model_path: Optional[Union[str, Path]] = None
) -> List[PredictionResult]:
    """
    Convenience function for batch prediction.
    
    Parameters
    ----------
    image_paths : List
        Paths to images
    model_path : str or Path, optional
        Path to trained model
        
    Returns
    -------
    List[PredictionResult]
        Prediction results
    """
    predictor = ImagePredictor(model_path=model_path)
    return predictor.predict_batch(image_paths)
