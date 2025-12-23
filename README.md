# AI Image Authenticity Checker

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.9+-blue.svg" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-red.svg" alt="PyTorch">
  <img src="https://img.shields.io/badge/CUDA-12.1-green.svg" alt="CUDA">
  <img src="https://img.shields.io/badge/Accuracy-99.45%25-brightgreen.svg" alt="Accuracy">
  <img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License">
</p>

A professional deep learning system for detecting AI-generated face images. Trained on the **[140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)** Kaggle dataset with **99.45% validation accuracy**.

## 🎯 Model Performance

| Metric | Value |
|--------|-------|
| **Validation Accuracy** | 99.45% |
| **Epochs Trained** | 57 |
| **Training Time** | 487.6 minutes |
| **Architecture** | EfficientNet-B4 |
| **GPU** | NVIDIA RTX 3060 (12GB) |

## 📊 Dataset

| Split | Real Images | Fake Images | Total |
|-------|-------------|-------------|-------|
| Train | 50,000 | 50,000 | 100,000 |
| Valid | 10,000 | 10,000 | 20,000 |
| Test | 10,000 | 10,000 | 20,000 |

**Source**: [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ai-image-authenticity-checker.git
cd ai-image-authenticity-checker

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows

# Install dependencies (with CUDA for RTX 3060)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### Predict on Images

```bash
# Single image prediction
python main.py predict --image path/to/image.jpg --model saved_models/best_model.pt

# Batch prediction
python main.py predict --directory ./images --output results.csv
```

### Start API Server

```bash
python main.py serve --port 8000
# API docs: http://localhost:8000/docs
```

## 🔧 Training

### GPU Training (Deep Learning)

```bash
# Train with GPU (auto-resumes from last checkpoint)
python train_gpu.py --epochs 100

# Test GPU setup
python train_gpu.py --test-gpu

# Start fresh (ignore checkpoints)
python train_gpu.py --epochs 100 --no-resume

# Quick test
python train_gpu.py --epochs 2 --max-samples 100 --dry-run
```

### Classical ML Training

```bash
python main.py train --algorithm rf      # Random Forest
python main.py train --algorithm xgboost # XGBoost
```

### Monitor Training

```bash
tensorboard --logdir outputs/logs/tensorboard
```

## 📁 Project Structure

```
ai-image-authenticity-checker/
├── data/
│   ├── train/real/          # 50K real training images
│   ├── train/fake/          # 50K fake training images
│   ├── valid/real/          # 10K real validation images
│   ├── valid/fake/          # 10K fake validation images
│   ├── test/real/           # 10K real test images
│   └── test/fake/           # 10K fake test images
├── saved_models/
│   ├── best_model.pt        # Best trained model (99.45% acc)
│   └── checkpoints/         # Training checkpoints
├── features/                # Feature extraction modules
├── model/                   # ML model definitions
├── inference/               # Prediction pipeline
├── train_gpu.py            # GPU training script
├── main.py                 # CLI entry point
├── config.py               # Configuration
└── requirements.txt
```

## 🌐 API Reference

### Predict Endpoint

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg"
```

### Response

```json
{
  "prediction": "AI-Generated",
  "confidence": 0.9823,
  "confidence_level": "high",
  "probabilities": {
    "Real": 0.0177,
    "AI-Generated": 0.9823
  }
}
```

## 🔬 Technical Details

### Model Architecture
- **Backbone**: EfficientNet-B4 (pretrained on ImageNet)
- **Fine-tuning**: Layer-wise learning rates (backbone: 1e-5, classifier: 1e-4)
- **Classification Head**: Dropout → Linear(1792→512) → ReLU → Dropout → Linear(512→2)

### Training Configuration
- **Optimizer**: AdamW (weight decay: 1e-5)
- **Scheduler**: Cosine Annealing with Warm Restarts
- **Batch Size**: 24 (with gradient accumulation = 2, effective batch = 48)
- **Mixed Precision**: FP16 (AMP) for faster training
- **Early Stopping**: Patience 15 epochs

### Feature Extraction (Classical ML)
- **FFT Analysis**: Frequency domain patterns, GAN fingerprint detection
- **ELA**: Error Level Analysis for compression artifacts
- **Texture**: LBP, GLCM, Gabor filters
- **Noise**: PRNU patterns, noise residuals

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

## 🙏 Acknowledgments

- [Kaggle 140k Real and Fake Faces Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
- [PyTorch](https://pytorch.org/) and [timm](https://github.com/huggingface/pytorch-image-models)
- NVIDIA for CUDA and RTX 3060 GPU
