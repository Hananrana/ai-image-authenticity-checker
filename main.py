#!/usr/bin/env python3
"""
AI Image Authenticity Checker - Main CLI
=========================================

Command-line interface for training and inference.

Usage:
    python main.py train --data-dir ./data
    python main.py predict --image path/to/image.jpg
    python main.py predict --directory ./images --output results.csv
    python main.py serve --port 8000
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from utils.logger import setup_logging, get_logger

logger = get_logger(__name__)


def train_command(args):
    """Handle train command."""
    from model.trainer import ModelTrainer
    
    logger.info("Starting training pipeline")
    
    trainer = ModelTrainer(
        real_dir=Path(args.real_dir) if args.real_dir else None,
        fake_dir=Path(args.fake_dir) if args.fake_dir else None,
        output_dir=Path(args.output_dir) if args.output_dir else None
    )
    
    results = trainer.run_full_pipeline(
        max_samples=args.max_samples,
        algorithm=args.algorithm
    )
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Algorithm: {results['algorithm']}")
    print(f"Test Accuracy: {results['test_accuracy']:.4f}")
    print(f"Test AUC: {results['test_auc']:.4f}")
    print(f"Model saved: {results['model_path']}")
    print("="*50)


def predict_command(args):
    """Handle predict command."""
    from inference.predict import ImagePredictor
    
    predictor = ImagePredictor(model_path=args.model)
    
    if args.image:
        # Single image prediction
        result = predictor.predict(args.image, return_features=args.explain)
        
        print("\n" + "="*50)
        print("PREDICTION RESULT")
        print("="*50)
        print(f"Image: {result.image_path}")
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.2%} ({result.confidence_level})")
        print(f"Probabilities:")
        for label, prob in result.probabilities.items():
            print(f"  {label}: {prob:.2%}")
        
        if result.feature_contributions:
            print(f"\nTop Contributing Features:")
            for feat, imp in result.feature_contributions.items():
                print(f"  {feat}: {imp:.4f}")
        print("="*50)
        
    elif args.directory:
        # Batch prediction
        input_dir = Path(args.directory)
        supported = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')
        images = [f for f in input_dir.iterdir() if f.suffix.lower() in supported]
        
        if not images:
            print(f"No images found in {input_dir}")
            return
        
        print(f"Processing {len(images)} images...")
        results = predictor.predict_batch(images)
        
        # Export results
        output_path = Path(args.output) if args.output else input_dir / "predictions.json"
        output_format = "csv" if output_path.suffix == ".csv" else "json"
        predictor.export_results(results, output_path, format=output_format)
        
        # Summary
        real_count = sum(1 for r in results if r.label == 0)
        fake_count = sum(1 for r in results if r.label == 1)
        error_count = sum(1 for r in results if r.label == -1)
        
        print("\n" + "="*50)
        print("BATCH PREDICTION SUMMARY")
        print("="*50)
        print(f"Total images: {len(results)}")
        print(f"Predicted Real: {real_count}")
        print(f"Predicted AI-Generated: {fake_count}")
        if error_count:
            print(f"Errors: {error_count}")
        print(f"Results saved to: {output_path}")
        print("="*50)


def serve_command(args):
    """Handle serve command (start API server)."""
    try:
        from api.server import create_app
        import uvicorn
        
        app = create_app(model_path=args.model)
        
        print(f"\nStarting API server on http://{args.host}:{args.port}")
        print("API Documentation: http://{args.host}:{args.port}/docs")
        
        uvicorn.run(app, host=args.host, port=args.port)
        
    except ImportError:
        print("FastAPI/Uvicorn not installed. Install with: pip install fastapi uvicorn")
        sys.exit(1)


def download_command(args):
    """Handle dataset download command."""
    from data.download_datasets import DatasetDownloader
    
    downloader = DatasetDownloader()
    
    if args.stats:
        stats = downloader.get_dataset_stats()
        print("\n" + "="*50)
        print("DATASET STATISTICS")
        print("="*50)
        print(f"Real images: {stats['real_images']}")
        print(f"Fake images: {stats['fake_images']}")
        print(f"Total: {stats['total']}")
        print(f"Real directory: {stats['real_dir']}")
        print(f"Fake directory: {stats['fake_dir']}")
        print("="*50)
    else:
        downloader.download_sample_dataset(args.num_real, args.num_fake)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="AI Image Authenticity Checker",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py download --sample
  python main.py train --algorithm rf
  python main.py predict --image photo.jpg
  python main.py predict --directory ./images --output results.csv
  python main.py serve --port 8000
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Download command
    download_parser = subparsers.add_parser("download", help="Download datasets")
    download_parser.add_argument("--sample", action="store_true", help="Download sample dataset")
    download_parser.add_argument("--num-real", type=int, default=100, help="Number of real images")
    download_parser.add_argument("--num-fake", type=int, default=100, help="Number of fake images")
    download_parser.add_argument("--stats", action="store_true", help="Show dataset statistics")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--real-dir", help="Directory with real images")
    train_parser.add_argument("--fake-dir", help="Directory with fake images")
    train_parser.add_argument("--output-dir", help="Output directory for model")
    train_parser.add_argument("--algorithm", default="rf", choices=["svm", "rf", "xgboost", "lightgbm"])
    train_parser.add_argument("--max-samples", type=int, help="Maximum samples to use")
    
    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Predict image authenticity")
    predict_parser.add_argument("--image", help="Single image to predict")
    predict_parser.add_argument("--directory", help="Directory of images")
    predict_parser.add_argument("--model", help="Path to trained model")
    predict_parser.add_argument("--output", help="Output file for batch results")
    predict_parser.add_argument("--explain", action="store_true", help="Include feature explanations")
    
    # Serve command
    serve_parser = subparsers.add_parser("serve", help="Start API server")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host to bind")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port to bind")
    serve_parser.add_argument("--model", help="Path to trained model")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="INFO")
    
    if args.command == "download":
        download_command(args)
    elif args.command == "train":
        train_command(args)
    elif args.command == "predict":
        predict_command(args)
    elif args.command == "serve":
        serve_command(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
