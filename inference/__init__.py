"""
Inference modules for AI Image Authenticity Detection.
"""

from .predict import ImagePredictor, predict_image, predict_batch

__all__ = [
    "ImagePredictor",
    "predict_image",
    "predict_batch"
]
