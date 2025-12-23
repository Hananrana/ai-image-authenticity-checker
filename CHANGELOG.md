# Changelog

All notable changes to this project will be documented in this file.

## [1.0.0] - 2025-12-23

### Added
- GPU-accelerated training with PyTorch and CUDA support
- Mixed precision training (AMP) for faster training on RTX 3060
- Automatic checkpoint resume functionality
- TensorBoard logging for training visualization
- Test evaluation script (`test_eval.py`)
- EfficientNet-B4 backbone with fine-tuning
- Gradient accumulation for effective batch size of 48
- Early stopping with patience of 15 epochs
- Cosine annealing learning rate scheduler

### Model Performance
- **Validation Accuracy**: 99.45%
- **Test Accuracy**: 99.39%
- **AUC-ROC**: 0.9999
- **Training Time**: 487.6 minutes (57 epochs)

### Dataset
- Trained on Kaggle 140k Real and Fake Faces dataset
- Train: 100,000 images (50K real, 50K fake)
- Validation: 20,000 images (10K real, 10K fake)
- Test: 20,000 images (10K real, 10K fake)

## [0.1.0] - 2025-12-22

### Added
- Initial project structure
- FFT-based frequency forensic feature extraction
- Error Level Analysis (ELA) features
- Texture analysis (LBP, GLCM, Gabor)
- Noise pattern analysis
- Classical ML classifiers (SVM, Random Forest, XGBoost)
- FastAPI REST server
- CLI interface
