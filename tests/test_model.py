"""
Tests for model modules.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_data():
    """Create sample training data."""
    np.random.seed(42)
    
    # Create simple separable data
    n_samples = 100
    n_features = 50
    
    # Class 0: centered around 0
    X0 = np.random.randn(n_samples // 2, n_features)
    
    # Class 1: centered around 2
    X1 = np.random.randn(n_samples // 2, n_features) + 2
    
    X = np.vstack([X0, X1])
    y = np.array([0] * (n_samples // 2) + [1] * (n_samples // 2))
    
    return X, y


class TestAIImageClassifier:
    """Tests for AIImageClassifier."""
    
    def test_classifier_initialization_rf(self):
        """Test Random Forest classifier initialization."""
        from model.classifier import AIImageClassifier
        
        clf = AIImageClassifier(algorithm="rf")
        assert clf.algorithm == "rf"
        assert not clf.is_fitted
    
    def test_classifier_initialization_svm(self):
        """Test SVM classifier initialization."""
        from model.classifier import AIImageClassifier
        
        clf = AIImageClassifier(algorithm="svm")
        assert clf.algorithm == "svm"
    
    def test_classifier_fit(self, sample_data):
        """Test classifier training."""
        from model.classifier import AIImageClassifier
        
        X, y = sample_data
        clf = AIImageClassifier(algorithm="rf")
        results = clf.fit(X, y)
        
        assert clf.is_fitted
        assert results.accuracy > 0.5
        assert results.auc_score > 0.5
    
    def test_classifier_predict(self, sample_data):
        """Test classifier prediction."""
        from model.classifier import AIImageClassifier
        
        X, y = sample_data
        clf = AIImageClassifier(algorithm="rf")
        clf.fit(X, y)
        
        predictions = clf.predict(X[:5])
        
        assert len(predictions) == 5
        assert all(p in [0, 1] for p in predictions)
    
    def test_classifier_predict_proba(self, sample_data):
        """Test classifier probability prediction."""
        from model.classifier import AIImageClassifier
        
        X, y = sample_data
        clf = AIImageClassifier(algorithm="rf")
        clf.fit(X, y)
        
        proba = clf.predict_proba(X[:5])
        
        assert proba.shape == (5, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)
    
    def test_classifier_save_load(self, sample_data):
        """Test classifier save and load."""
        from model.classifier import AIImageClassifier
        
        X, y = sample_data
        clf = AIImageClassifier(algorithm="rf")
        clf.fit(X, y)
        
        with tempfile.NamedTemporaryFile(suffix='.joblib', delete=False) as f:
            clf.save(f.name)
            
            loaded_clf = AIImageClassifier.load(f.name)
            
            assert loaded_clf.is_fitted
            assert loaded_clf.algorithm == "rf"
            
            # Predictions should match
            orig_pred = clf.predict(X[:5])
            loaded_pred = loaded_clf.predict(X[:5])
            
            assert np.array_equal(orig_pred, loaded_pred)
        
        Path(f.name).unlink(missing_ok=True)


class TestEnsembleClassifier:
    """Tests for EnsembleClassifier."""
    
    def test_ensemble_initialization(self):
        """Test ensemble initialization."""
        from model.ensemble import EnsembleClassifier
        
        ensemble = EnsembleClassifier(include_algorithms=["rf", "svm"])
        assert len(ensemble.include_algorithms) == 2
    
    def test_ensemble_fit(self, sample_data):
        """Test ensemble training."""
        from model.ensemble import EnsembleClassifier
        
        X, y = sample_data
        ensemble = EnsembleClassifier(include_algorithms=["rf", "svm"])
        ensemble.fit(X, y)
        
        assert ensemble.is_fitted
        assert len(ensemble.classifiers) >= 1
    
    def test_ensemble_predict(self, sample_data):
        """Test ensemble prediction."""
        from model.ensemble import EnsembleClassifier
        
        X, y = sample_data
        ensemble = EnsembleClassifier(include_algorithms=["rf"])
        ensemble.fit(X, y)
        
        predictions = ensemble.predict(X[:5])
        
        assert len(predictions) == 5


class TestModelTrainer:
    """Tests for ModelTrainer."""
    
    def test_trainer_initialization(self):
        """Test trainer initialization."""
        from model.trainer import ModelTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(
                real_dir=Path(tmpdir),
                fake_dir=Path(tmpdir),
                output_dir=Path(tmpdir)
            )
            
            assert trainer.real_dir.exists()
            assert trainer.fake_dir.exists()
    
    def test_trainer_split_data(self, sample_data):
        """Test data splitting."""
        from model.trainer import ModelTrainer
        
        with tempfile.TemporaryDirectory() as tmpdir:
            trainer = ModelTrainer(
                real_dir=Path(tmpdir),
                fake_dir=Path(tmpdir)
            )
            
            X, y = sample_data
            trainer.split_data(X, y)
            
            assert trainer.X_train is not None
            assert trainer.X_val is not None
            assert trainer.X_test is not None
            
            total = len(trainer.X_train) + len(trainer.X_val) + len(trainer.X_test)
            assert total == len(X)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
