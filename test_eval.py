#!/usr/bin/env python3
"""
Test Set Evaluation Script
==========================

Evaluates the trained model on the test dataset.

Usage:
    python test_eval.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, confusion_matrix, classification_report
)

from config import DL_CONFIG, TEST_DIR, MODEL_DIR
from train_gpu import FaceAuthenticityModel, FaceAuthenticityDataset, get_val_transforms
from utils.logger import setup_logging, get_logger

setup_logging(level="INFO")
logger = get_logger(__name__)


def evaluate_test_set():
    """
    Evaluate the best model on the test dataset.
    Prints accuracy, AUC, precision, recall, F1-score, and confusion matrix.
    """
    print("\n" + "="*60)
    print("TEST SET EVALUATION")
    print("="*60)
    
    # Check for best model
    best_model_path = MODEL_DIR / "best_model.pt"
    if not best_model_path.exists():
        print(f"ERROR: No trained model found at {best_model_path}")
        print("Please train a model first: python train_gpu.py --epochs 50")
        return
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Load model
    print(f"Loading model from: {best_model_path}")
    checkpoint = torch.load(best_model_path, map_location=device, weights_only=False)
    
    model = FaceAuthenticityModel(
        backbone=checkpoint['config']['backbone'],
        num_classes=checkpoint['config']['num_classes'],
        dropout_rate=checkpoint['config']['dropout_rate'],
        pretrained=False
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model trained for {checkpoint['epoch']} epochs")
    print(f"Best validation accuracy: {checkpoint['best_val_acc']:.2%}")
    
    # Load test dataset
    print(f"\nLoading test data from: {TEST_DIR}")
    test_dataset = FaceAuthenticityDataset(
        TEST_DIR,
        transform=get_val_transforms()
    )
    
    if len(test_dataset) == 0:
        print(f"ERROR: No test images found in {TEST_DIR}")
        print(f"Expected structure: {TEST_DIR}/real and {TEST_DIR}/fake")
        return
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=DL_CONFIG.batch_size * 2,
        shuffle=False,
        num_workers=DL_CONFIG.num_workers,
        pin_memory=True
    )
    
    print(f"Test samples: {len(test_dataset)} (Real: {test_dataset.real_count}, Fake: {test_dataset.fake_count})")
    
    # Evaluate
    print("\nEvaluating...")
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing"):
            images = images.to(device)
            labels = labels.to(device)
            
            with autocast(enabled=True):
                outputs = model(images)
            
            probs = outputs.softmax(dim=1)
            preds = probs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    
    # Print results
    print("\n" + "="*60)
    print("TEST SET RESULTS")
    print("="*60)
    print(f"")
    print(f"  Test Samples: {len(test_dataset)}")
    print(f"")
    print(f"  Accuracy:  {accuracy:.4f} ({accuracy:.2%})")
    print(f"  AUC-ROC:   {auc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"")
    print("Confusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Real    Fake")
    print(f"  Actual Real    {cm[0][0]:5d}   {cm[0][1]:5d}")
    print(f"  Actual Fake    {cm[1][0]:5d}   {cm[1][1]:5d}")
    print(f"")
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['Real', 'AI-Generated']))
    print("="*60 + "\n")


if __name__ == "__main__":
    evaluate_test_set()
