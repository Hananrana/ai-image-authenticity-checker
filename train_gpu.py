#!/usr/bin/env python3
"""
GPU-Accelerated Deep Learning Training
=======================================

Complete PyTorch training pipeline optimized for NVIDIA RTX 3060.
Supports mixed precision training, gradient accumulation, and TensorBoard logging.

Usage:
    python train_gpu.py                          # Full training
    python train_gpu.py --test-gpu               # Test GPU availability
    python train_gpu.py --epochs 10 --dry-run    # Quick test run
    python train_gpu.py --resume checkpoint.pt   # Resume training
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any, Union, List

import numpy as np
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter

# Torchvision imports
from torchvision import transforms
from PIL import Image

# Optional: timm for advanced models
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    timm = None  # type: ignore
    TIMM_AVAILABLE = False

# Project imports
from config import (
    DL_CONFIG, TRAIN_DIR, VALID_DIR, TEST_DIR,
    MODEL_DIR, CHECKPOINT_DIR, TENSORBOARD_DIR
)
from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


# =============================================================================
# DATASET
# =============================================================================

class FaceAuthenticityDataset(Dataset):
    """
    Dataset for real vs fake face classification.
    Expects folder structure: split_dir/real/*.jpg, split_dir/fake/*.jpg
    """

    SUPPORTED_FORMATS = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    def __init__(
        self,
        root_dir: Path,
        transform: Optional[transforms.Compose] = None,
        max_samples: Optional[int] = None
    ):
        self.root_dir = Path(root_dir)
        self.transform = transform or self._default_transform()

        # Load image paths and labels
        self.samples = []

        real_dir = self.root_dir / "real"
        fake_dir = self.root_dir / "fake"

        # Load real images (label=0)
        if real_dir.exists():
            real_images = [f for f in real_dir.iterdir()
                          if f.suffix.lower() in self.SUPPORTED_FORMATS]
            for img_path in real_images:
                self.samples.append((img_path, 0))

        # Load fake images (label=1)
        if fake_dir.exists():
            fake_images = [f for f in fake_dir.iterdir()
                          if f.suffix.lower() in self.SUPPORTED_FORMATS]
            for img_path in fake_images:
                self.samples.append((img_path, 1))

        # Limit samples if specified
        if max_samples and len(self.samples) > max_samples:
            np.random.shuffle(self.samples)
            self.samples = self.samples[:max_samples]

        # Count per class
        self.real_count = sum(1 for _, label in self.samples if label == 0)
        self.fake_count = sum(1 for _, label in self.samples if label == 1)

        logger.info(f"Loaded {len(self.samples)} images from {root_dir}")
        logger.info(f"  Real: {self.real_count}, Fake: {self.fake_count}")

    @staticmethod
    def _default_transform() -> transforms.Compose:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.warning(f"Error loading {img_path}: {e}")
            # Return a black image as fallback
            return torch.zeros(3, 224, 224), label


def get_train_transforms(augment_prob: float = 0.5) -> transforms.Compose:
    """Training transforms with augmentation."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(
            brightness=0.2 * augment_prob,
            contrast=0.2 * augment_prob,
            saturation=0.2 * augment_prob,
            hue=0.1 * augment_prob
        ),
        transforms.RandomGrayscale(p=0.05 * augment_prob),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        transforms.RandomErasing(p=0.1 * augment_prob)
    ])


def get_val_transforms() -> transforms.Compose:
    """Validation/test transforms (no augmentation)."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


# =============================================================================
# MODEL
# =============================================================================

class FaceAuthenticityModel(nn.Module):
    """
    Face authenticity classifier using pretrained backbone.
    """

    def __init__(
        self,
        backbone: str = "efficientnet_b4",
        num_classes: int = 2,
        dropout_rate: float = 0.3,
        pretrained: bool = True
    ):
        super().__init__()
        self.backbone_name = backbone

        if TIMM_AVAILABLE and "efficientnet" in backbone:
            # Use timm for EfficientNet
            self.backbone = timm.create_model(
                backbone,
                pretrained=pretrained,
                num_classes=0  # Remove classifier
            )
            in_features = self.backbone.num_features
        else:
            # Fallback to torchvision
            from torchvision import models
            if "efficientnet" in backbone:
                self.backbone = models.efficientnet_b0(
                    weights='IMAGENET1K_V1' if pretrained else None
                )
                in_features = self.backbone.classifier[1].in_features
                self.backbone.classifier = nn.Identity()
            else:
                self.backbone = models.resnet50(
                    weights='IMAGENET1K_V1' if pretrained else None
                )
                in_features = self.backbone.fc.in_features
                self.backbone.fc = nn.Identity()

        # Classification head
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        return self.classifier(features)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features without classification."""
        return self.backbone(x)


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_score: float) -> bool:
        if self.best_score is None:
            self.best_score = val_score
        elif val_score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        else:
            self.best_score = val_score
            self.counter = 0

        return self.should_stop


class MetricsTracker:
    """Track and compute training metrics."""

    def __init__(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0
        self.predictions: List[float] = []
        self.labels: List[int] = []

    def reset(self):
        self.loss_sum = 0.0
        self.correct = 0
        self.total = 0
        self.predictions = []
        self.labels = []

    def update(self, loss: float, preds: torch.Tensor, labels: torch.Tensor):
        self.loss_sum += loss
        self.correct += (preds.argmax(dim=1) == labels).sum().item()
        self.total += labels.size(0)
        self.predictions.extend(preds.softmax(dim=1)[:, 1].cpu().numpy().tolist())
        self.labels.extend(labels.cpu().numpy().tolist())

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def avg_loss(self) -> float:
        return self.loss_sum / max(self.total, 1)

    def compute_auc(self) -> float:
        from sklearn.metrics import roc_auc_score
        try:
            return float(roc_auc_score(self.labels, self.predictions))
        except ValueError:
            return 0.5


# =============================================================================
# TRAINER
# =============================================================================

class GPUTrainer:
    """
    GPU-accelerated trainer with mixed precision support.
    """

    def __init__(
        self,
        config: Any = None,
        model: Optional[nn.Module] = None,
        experiment_name: Optional[str] = None
    ):
        self.config = config or DL_CONFIG
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Setup device
        self.device = self._setup_device()

        # Initialize model
        self.model = model or FaceAuthenticityModel(
            backbone=self.config.backbone,
            num_classes=self.config.num_classes,
            dropout_rate=self.config.dropout_rate,
            pretrained=self.config.pretrained
        )
        self.model = self.model.to(self.device)

        # Mixed precision scaler
        self.scaler = GradScaler(enabled=self.config.use_amp)

        # Optimizer and scheduler (initialized during training)
        self.optimizer = None
        self.scheduler = None

        # TensorBoard
        self.writer = SummaryWriter(
            log_dir=TENSORBOARD_DIR / self.experiment_name
        )

        # Training state
        self.current_epoch = 0
        self.best_val_acc = 0.0
        self.global_step = 0

        logger.info(f"Initialized GPUTrainer on {self.device}")
        logger.info(f"Model: {self.config.backbone}, AMP: {self.config.use_amp}")

    def _setup_device(self) -> torch.device:
        """Setup CUDA device with error handling."""
        if self.config.device == "cuda" and torch.cuda.is_available():
            device = torch.device("cuda")

            # Log GPU info
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Using GPU: {gpu_name} ({gpu_memory:.1f} GB)")

            # Enable cuDNN benchmarking for faster training
            torch.backends.cudnn.benchmark = True

            return device
        else:
            logger.warning("CUDA not available, using CPU")
            return torch.device("cpu")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with layer-wise learning rates."""
        # Lower LR for pretrained backbone, higher for classifier
        backbone_params = list(self.model.backbone.parameters())
        classifier_params = list(self.model.classifier.parameters())

        param_groups = [
            {'params': backbone_params, 'lr': self.config.learning_rate * 0.1},
            {'params': classifier_params, 'lr': self.config.learning_rate}
        ]

        return optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay
        )

    def _create_scheduler(self, num_training_steps: int):
        """Create learning rate scheduler with warmup."""
        warmup_steps = self.config.warmup_epochs * (num_training_steps // self.config.num_epochs)

        if self.config.scheduler == "cosine":
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=max(1, num_training_steps - warmup_steps),
                T_mult=1
            )
        elif self.config.scheduler == "plateau":
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='max',
                factor=0.5,
                patience=5
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=10,
                gamma=0.1
            )

        return scheduler

    def create_dataloaders(
        self,
        train_dir: Path = TRAIN_DIR,
        val_dir: Path = VALID_DIR,
        max_samples: Optional[int] = None
    ) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation dataloaders."""

        train_dataset = FaceAuthenticityDataset(
            train_dir,
            transform=get_train_transforms(self.config.augmentation_prob),
            max_samples=max_samples
        )

        val_dataset = FaceAuthenticityDataset(
            val_dir,
            transform=get_val_transforms(),
            max_samples=max_samples // 5 if max_samples else None
        )

        # Adjust batch size for small datasets
        train_batch_size = min(self.config.batch_size, len(train_dataset)) if len(train_dataset) > 0 else 1
        val_batch_size = min(self.config.batch_size * 2, len(val_dataset)) if len(val_dataset) > 0 else 1

        # Only drop last if we have enough samples for multiple batches
        drop_last = len(train_dataset) > train_batch_size

        train_loader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
            drop_last=drop_last
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None
        )

        return train_loader, val_loader

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        metrics = MetricsTracker()

        accumulation_steps = self.config.gradient_accumulation_steps
        self.optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")

        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            # Forward pass with mixed precision
            with autocast(enabled=self.config.use_amp):
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                loss = loss / accumulation_steps  # Scale loss for accumulation

            # Backward pass with gradient scaling
            self.scaler.scale(loss).backward()

            # Update weights after accumulation steps
            if (batch_idx + 1) % accumulation_steps == 0:
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.global_step += 1

            # Track metrics
            metrics.update(loss.item() * accumulation_steps, outputs.detach(), labels)

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{metrics.avg_loss:.4f}',
                'acc': f'{metrics.accuracy:.2%}'
            })

        return {
            'train_loss': metrics.avg_loss,
            'train_acc': metrics.accuracy
        }

    @torch.no_grad()
    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module
    ) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        metrics = MetricsTracker()

        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(enabled=self.config.use_amp):
                outputs = self.model(images)
                loss = criterion(outputs, labels)

            metrics.update(loss.item(), outputs, labels)

        return {
            'val_loss': metrics.avg_loss,
            'val_acc': metrics.accuracy,
            'val_auc': metrics.compute_auc()
        }

    def save_checkpoint(
        self,
        is_best: bool = False,
        filename: Optional[str] = None
    ):
        """Save model checkpoint."""
        filename = filename or f"checkpoint_epoch_{self.current_epoch}.pt"
        checkpoint_path = CHECKPOINT_DIR / filename

        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'best_val_acc': self.best_val_acc,
            'config': {
                'backbone': self.config.backbone,
                'num_classes': self.config.num_classes,
                'dropout_rate': self.config.dropout_rate
            }
        }

        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")

        if is_best:
            best_path = MODEL_DIR / "best_model.pt"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best model: {best_path}")

    def load_checkpoint(self, checkpoint_path: Path, resume_training: bool = True):
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            resume_training: If True, resume from next epoch; if False, start from epoch 0
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if resume_training:
            # Resume from next epoch
            self.current_epoch = checkpoint['epoch'] + 1
            self.best_val_acc = checkpoint.get('best_val_acc', 0.0)
            logger.info(f"Resuming training from epoch {self.current_epoch}")
        else:
            self.current_epoch = 0
            self.best_val_acc = 0.0
            logger.info(f"Loaded weights from epoch {checkpoint['epoch']} (starting fresh)")

        if self.optimizer and 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

    @staticmethod
    def find_latest_checkpoint(checkpoint_dir: Path = CHECKPOINT_DIR) -> Optional[Path]:
        """
        Find the latest checkpoint in the checkpoint directory.
        
        Returns:
            Path to the latest checkpoint, or None if no checkpoints found
        """
        if not checkpoint_dir.exists():
            return None
        
        # Find all checkpoint files
        checkpoints = list(checkpoint_dir.glob("checkpoint_epoch_*.pt"))
        
        if not checkpoints:
            return None
        
        # Extract epoch numbers and find the latest
        def get_epoch(cp: Path) -> int:
            try:
                # Extract number from "checkpoint_epoch_X.pt"
                name = cp.stem  # "checkpoint_epoch_5"
                epoch = int(name.split('_')[-1])
                return epoch
            except (ValueError, IndexError):
                return -1
        
        # Sort by epoch number and get the latest
        checkpoints.sort(key=get_epoch, reverse=True)
        latest = checkpoints[0]
        
        logger.info(f"Found {len(checkpoints)} checkpoints, latest: {latest.name}")
        return latest


    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: Optional[int] = None
    ) -> Dict[str, Any]:
        """Full training loop."""
        num_epochs = num_epochs or self.config.num_epochs

        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        num_training_steps = num_epochs * len(train_loader)
        self.scheduler = self._create_scheduler(num_training_steps)

        # Loss function with class weights if imbalanced
        criterion = nn.CrossEntropyLoss()

        # Early stopping
        early_stopping = EarlyStopping(
            patience=self.config.early_stopping_patience,
            min_delta=self.config.min_delta
        )

        # Training history
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [], 'val_auc': []
        }

        start_time = time.time()

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_metrics = self.train_epoch(train_loader, criterion)

            # Validate
            val_metrics = self.validate(val_loader, criterion)

            # Update history
            for key, value in {**train_metrics, **val_metrics}.items():
                history[key].append(value)

            # Log to TensorBoard
            for key, value in {**train_metrics, **val_metrics}.items():
                self.writer.add_scalar(key, value, epoch)
            self.writer.add_scalar('lr', self.optimizer.param_groups[0]['lr'], epoch)

            # Update learning rate (for plateau scheduler)
            if self.config.scheduler == "plateau":
                self.scheduler.step(val_metrics['val_acc'])
            else:
                self.scheduler.step()

            epoch_time = time.time() - epoch_start

            # Log epoch results
            logger.info(
                f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) - "
                f"Train Loss: {train_metrics['train_loss']:.4f}, "
                f"Train Acc: {train_metrics['train_acc']:.2%}, "
                f"Val Loss: {val_metrics['val_loss']:.4f}, "
                f"Val Acc: {val_metrics['val_acc']:.2%}, "
                f"Val AUC: {val_metrics['val_auc']:.4f}"
            )

            # Save best model
            is_best = val_metrics['val_acc'] > self.best_val_acc
            if is_best:
                self.best_val_acc = val_metrics['val_acc']

            # Save checkpoint
            if (epoch + 1) % self.config.save_every_n_epochs == 0 or is_best:
                self.save_checkpoint(is_best=is_best)

            # Early stopping
            if early_stopping(val_metrics['val_acc']):
                logger.info(f"Early stopping triggered at epoch {epoch}")
                break

        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/60:.1f} minutes")
        logger.info(f"Best validation accuracy: {self.best_val_acc:.2%}")

        self.writer.close()

        return {
            'history': history,
            'best_val_acc': self.best_val_acc,
            'total_time': total_time,
            'epochs_trained': self.current_epoch + 1
        }


# =============================================================================
# GPU TEST
# =============================================================================

def test_gpu():
    """Test GPU availability and capabilities."""
    print("\n" + "="*60)
    print("GPU SYSTEM TEST")
    print("="*60)

    print(f"\nPyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"GPU count: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"\nGPU {i}: {props.name}")
            print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
            print(f"  CUDA Cores: ~{props.multi_processor_count * 128}")  # Approximate
            print(f"  Compute Capability: {props.major}.{props.minor}")

        # Test memory allocation
        print("\nTesting GPU memory allocation...")
        try:
            test_tensor = torch.randn(1000, 1000, device='cuda')
            print(f"  Allocated {test_tensor.element_size() * test_tensor.numel() / 1024**2:.1f} MB")
            del test_tensor
            torch.cuda.empty_cache()
            print("  Memory test: PASSED")
        except Exception as e:
            print(f"  Memory test: FAILED - {e}")

        # Test model loading
        print("\nTesting model creation...")
        try:
            model = FaceAuthenticityModel(backbone="efficientnet_b4", pretrained=False)
            model = model.to('cuda')
            test_input = torch.randn(1, 3, 224, 224, device='cuda')
            with torch.no_grad():
                output = model(test_input)
            print(f"  Model output shape: {output.shape}")
            print("  Model test: PASSED")

            # Estimate batch size
            del model, test_input, output
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  Model test: FAILED - {e}")

        # Test mixed precision
        print("\nTesting mixed precision (AMP)...")
        try:
            model = FaceAuthenticityModel(backbone="efficientnet_b4", pretrained=False).cuda()
            test_input = torch.randn(4, 3, 224, 224, device='cuda')
            with autocast(enabled=True):
                output = model(test_input)
            print(f"  AMP output dtype: {output.dtype}")
            print("  AMP test: PASSED")
        except Exception as e:
            print(f"  AMP test: FAILED - {e}")

    else:
        print("\nWARNING: No CUDA-capable GPU detected!")
        print("Training will use CPU (much slower)")

    print("\n" + "="*60)
    print("GPU TEST COMPLETE")
    print("="*60 + "\n")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="GPU-accelerated training for AI image authenticity detection",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--test-gpu', action='store_true',
                       help='Test GPU availability and exit')
    parser.add_argument('--dry-run', action='store_true',
                       help='Quick test with minimal samples')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default from config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--max-samples', type=int, default=None,
                       help='Maximum samples to use')
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to specific checkpoint to resume from')
    parser.add_argument('--auto-resume', action='store_true', default=True,
                       help='Automatically resume from latest checkpoint (default: True)')
    parser.add_argument('--no-resume', action='store_true',
                       help='Start fresh training, ignore existing checkpoints')
    parser.add_argument('--test-only', action='store_true',
                       help='Evaluate model on test set only (no training)')
    parser.add_argument('--backbone', type=str, default=None,
                       help='Model backbone (efficientnet_b0/b4, resnet50)')
    parser.add_argument('--name', type=str, default=None,
                       help='Experiment name')

    args = parser.parse_args()

    # Setup logging
    setup_logging(level="INFO")

    # GPU test mode
    if args.test_gpu:
        test_gpu()
        return

    # Test-only mode: evaluate on test set
    if args.test_only:
        evaluate_test_set()
        return

    # Override config if needed
    if args.batch_size:
        DL_CONFIG.batch_size = args.batch_size
    if args.lr:
        DL_CONFIG.learning_rate = args.lr
    if args.backbone:
        DL_CONFIG.backbone = args.backbone

    # Dry run settings
    max_samples = args.max_samples
    num_epochs = args.epochs or DL_CONFIG.num_epochs

    if args.dry_run:
        max_samples = max_samples or 100
        num_epochs = min(num_epochs, 2)
        logger.info("DRY RUN MODE: Using minimal samples and epochs")

    # Create trainer
    trainer = GPUTrainer(
        config=DL_CONFIG,
        experiment_name=args.name
    )

    # Handle checkpoint resumption
    if args.resume:
        # Explicit checkpoint path provided
        trainer.load_checkpoint(Path(args.resume), resume_training=True)
        logger.info(f"Resuming from specified checkpoint: {args.resume}")
    elif args.no_resume:
        # User explicitly wants fresh start
        logger.info("Starting fresh training (--no-resume specified)")
    elif args.auto_resume:
        # Auto-resume: find latest checkpoint
        latest_checkpoint = GPUTrainer.find_latest_checkpoint()
        if latest_checkpoint:
            trainer.load_checkpoint(latest_checkpoint, resume_training=True)
        else:
            logger.info("No existing checkpoints found, starting fresh training")

    # Create dataloaders
    train_loader, val_loader = trainer.create_dataloaders(
        max_samples=max_samples
    )

    if len(train_loader) == 0:
        logger.error("No training data found! Check your data directory.")
        logger.error(f"Expected structure: {TRAIN_DIR}/real and {TRAIN_DIR}/fake")
        return

    # Train
    results = trainer.train(
        train_loader,
        val_loader,
        num_epochs=num_epochs
    )

    # Final summary
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"Best Validation Accuracy: {results['best_val_acc']:.2%}")
    print(f"Epochs Trained: {results['epochs_trained']}")
    print(f"Total Time: {results['total_time']/60:.1f} minutes")
    print(f"Best Model: {MODEL_DIR / 'best_model.pt'}")
    print(f"TensorBoard: tensorboard --logdir {TENSORBOARD_DIR}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
