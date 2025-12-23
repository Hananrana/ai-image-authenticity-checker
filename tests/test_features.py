"""
Tests for feature extraction modules.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
from PIL import Image
import sys

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture
def sample_image():
    """Create a sample test image."""
    img = np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)
    
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
        Image.fromarray(img).save(f.name)
        yield f.name
    
    Path(f.name).unlink(missing_ok=True)


@pytest.fixture
def sample_image_array():
    """Create a sample test image array."""
    return np.random.randint(0, 256, (256, 256, 3), dtype=np.uint8)


class TestFFTFeatures:
    """Tests for FFT feature extraction."""
    
    def test_fft_extractor_initialization(self):
        """Test FFT extractor initializes correctly."""
        from features.fft_features import FFTFeatureExtractor
        
        extractor = FFTFeatureExtractor()
        assert extractor.image_size == 256
        assert extractor.low_freq_radius == 32
        assert extractor.num_bands == 8
    
    def test_fft_extract_from_path(self, sample_image):
        """Test FFT extraction from file path."""
        from features.fft_features import FFTFeatureExtractor
        
        extractor = FFTFeatureExtractor()
        features = extractor.extract(sample_image)
        
        assert hasattr(features, 'low_freq_energy')
        assert hasattr(features, 'high_freq_energy')
        assert hasattr(features, 'spectral_ratio')
        assert hasattr(features, 'gan_fingerprint_score')
    
    def test_fft_extract_from_array(self, sample_image_array):
        """Test FFT extraction from numpy array."""
        from features.fft_features import FFTFeatureExtractor
        
        extractor = FFTFeatureExtractor()
        features = extractor.extract(sample_image_array)
        
        assert features.low_freq_energy > 0
        assert len(features.azimuthal_profile) == 32
        assert len(features.band_energies) == 8
    
    def test_fft_to_array(self, sample_image):
        """Test FFT features conversion to array."""
        from features.fft_features import FFTFeatureExtractor
        
        extractor = FFTFeatureExtractor()
        features = extractor.extract(sample_image)
        array = features.to_array()
        
        assert isinstance(array, np.ndarray)
        assert len(array) > 0
    
    def test_convenience_function(self, sample_image):
        """Test convenience extraction function."""
        from features.fft_features import extract_fft_features
        
        features = extract_fft_features(sample_image)
        
        assert isinstance(features, np.ndarray)
        assert len(features) > 0


class TestELAFeatures:
    """Tests for ELA feature extraction."""
    
    def test_ela_extractor_initialization(self):
        """Test ELA extractor initializes correctly."""
        from features.ela_features import ELAFeatureExtractor
        
        extractor = ELAFeatureExtractor()
        assert extractor.quality_levels == [90, 75, 50]
        assert extractor.block_size == 8
    
    def test_ela_extract_from_path(self, sample_image):
        """Test ELA extraction from file path."""
        from features.ela_features import ELAFeatureExtractor
        
        extractor = ELAFeatureExtractor()
        features = extractor.extract(sample_image)
        
        assert hasattr(features, 'ela_mean')
        assert hasattr(features, 'ela_std')
        assert hasattr(features, 'ela_entropy')
    
    def test_ela_to_array(self, sample_image):
        """Test ELA features conversion to array."""
        from features.ela_features import ELAFeatureExtractor
        
        extractor = ELAFeatureExtractor()
        features = extractor.extract(sample_image)
        array = features.to_array()
        
        assert isinstance(array, np.ndarray)
        assert len(array) > 0


class TestTextureFeatures:
    """Tests for texture feature extraction."""
    
    def test_texture_extractor_initialization(self):
        """Test texture extractor initializes correctly."""
        from features.texture_features import TextureFeatureExtractor
        
        extractor = TextureFeatureExtractor()
        assert extractor.lbp_radius == 3
        assert extractor.lbp_n_points == 24
    
    def test_texture_extract_from_path(self, sample_image):
        """Test texture extraction from file path."""
        from features.texture_features import TextureFeatureExtractor
        
        extractor = TextureFeatureExtractor()
        features = extractor.extract(sample_image)
        
        assert hasattr(features, 'lbp_histogram')
        assert hasattr(features, 'glcm_contrast')
        assert hasattr(features, 'gabor_means')
    
    def test_texture_to_array(self, sample_image):
        """Test texture features conversion to array."""
        from features.texture_features import TextureFeatureExtractor
        
        extractor = TextureFeatureExtractor()
        features = extractor.extract(sample_image)
        array = features.to_array()
        
        assert isinstance(array, np.ndarray)
        assert len(array) > 0


class TestNoiseFeatures:
    """Tests for noise feature extraction."""
    
    def test_noise_extractor_initialization(self):
        """Test noise extractor initializes correctly."""
        from features.noise_features import NoiseFeatureExtractor
        
        extractor = NoiseFeatureExtractor()
        assert extractor.patch_size == 64
        assert extractor.num_patches == 16
    
    def test_noise_extract_from_path(self, sample_image):
        """Test noise extraction from file path."""
        from features.noise_features import NoiseFeatureExtractor
        
        extractor = NoiseFeatureExtractor()
        features = extractor.extract(sample_image)
        
        assert hasattr(features, 'noise_mean')
        assert hasattr(features, 'noise_std')
        assert hasattr(features, 'snr_estimate')


class TestFeatureFusion:
    """Tests for feature fusion module."""
    
    def test_fusion_initialization(self):
        """Test fusion module initializes correctly."""
        from features.feature_fusion import FeatureFusion
        
        fusion = FeatureFusion(include_deep=False)
        assert fusion.fft_extractor is not None
        assert fusion.ela_extractor is not None
    
    def test_fusion_extract(self, sample_image):
        """Test combined feature extraction."""
        from features.feature_fusion import FeatureFusion
        
        fusion = FeatureFusion(include_deep=False)
        features = fusion.extract(sample_image)
        
        assert features.fft_features is not None
        assert features.ela_features is not None
        assert features.texture_features is not None
        assert features.noise_features is not None
    
    def test_fusion_to_array(self, sample_image):
        """Test fusion features conversion to array."""
        from features.feature_fusion import FeatureFusion
        
        fusion = FeatureFusion(include_deep=False)
        features = fusion.extract(sample_image)
        array = features.to_array(include_deep=False)
        
        assert isinstance(array, np.ndarray)
        assert len(array) > 100  # Should have many features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
