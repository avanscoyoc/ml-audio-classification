"""Test audio preprocessing functionality."""

import numpy as np
import pytest
from unittest.mock import Mock, AsyncMock, patch
from pathlib import Path

from ml_audio_classification.data.audio_preprocessor import AudioPreprocessor
from ml_audio_classification.core.exceptions import AudioProcessingError


class TestAudioPreprocessor:
    """Test audio preprocessing functionality."""
    
    @pytest.fixture
    def audio_preprocessor(self):
        """Create audio preprocessor instance."""
        return AudioPreprocessor()
    
    @pytest.fixture
    def mock_audio_data(self):
        """Create mock audio data."""
        # 1 second of audio at 22050 Hz
        return np.random.randn(22050).astype(np.float32)
    
    def test_init(self, audio_preprocessor):
        """Test preprocessor initialization."""
        assert audio_preprocessor.sample_rate == 22050
        assert audio_preprocessor.n_mfcc == 13
        assert audio_preprocessor.n_fft == 2048
    
    @patch('librosa.load')
    async def test_load_audio_file(self, mock_load, audio_preprocessor, mock_audio_data):
        """Test audio file loading."""
        mock_load.return_value = (mock_audio_data, 22050)
        
        result = await audio_preprocessor.load_audio_file(Path("test.wav"))
        
        assert isinstance(result, np.ndarray)
        assert len(result) == len(mock_audio_data)
        mock_load.assert_called_once()
    
    @patch('librosa.load')
    async def test_load_audio_file_error(self, mock_load, audio_preprocessor):
        """Test audio file loading error handling."""
        mock_load.side_effect = Exception("File not found")
        
        with pytest.raises(AudioProcessingError):
            await audio_preprocessor.load_audio_file(Path("nonexistent.wav"))
    
    def test_extract_mfcc_features(self, audio_preprocessor, mock_audio_data):
        """Test MFCC feature extraction."""
        features = audio_preprocessor.extract_mfcc_features(mock_audio_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[0] == 13  # n_mfcc
        assert features.ndim == 2
    
    def test_extract_mfcc_features_with_derivatives(self, audio_preprocessor, mock_audio_data):
        """Test MFCC feature extraction with derivatives."""
        features = audio_preprocessor.extract_mfcc_features(
            mock_audio_data, 
            include_derivatives=True
        )
        
        # 13 MFCC + 13 delta + 13 delta-delta = 39 features
        assert features.shape[0] == 39
    
    def test_create_spectrogram(self, audio_preprocessor, mock_audio_data):
        """Test spectrogram creation."""
        spectrogram = audio_preprocessor.create_spectrogram(mock_audio_data)
        
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2
        assert spectrogram.shape[0] == 128  # n_mels
    
    def test_normalize_audio(self, audio_preprocessor):
        """Test audio normalization."""
        # Create audio with known range
        audio = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        normalized = audio_preprocessor.normalize_audio(audio)
        
        # Should be normalized to [-1, 1] range
        assert np.max(normalized) <= 1.0
        assert np.min(normalized) >= -1.0
    
    def test_resample_audio(self, audio_preprocessor):
        """Test audio resampling."""
        # 1 second at 44100 Hz
        original_audio = np.random.randn(44100)
        
        resampled = audio_preprocessor.resample_audio(original_audio, 44100, 22050)
        
        # Should be half the length when downsampling from 44100 to 22050
        expected_length = len(original_audio) // 2
        assert abs(len(resampled) - expected_length) <= 1  # Allow for rounding
    
    def test_pad_or_truncate(self, audio_preprocessor):
        """Test audio padding and truncation."""
        target_length = 22050  # 1 second at 22050 Hz
        
        # Test truncation
        long_audio = np.random.randn(44100)  # 2 seconds
        truncated = audio_preprocessor.pad_or_truncate(long_audio, target_length)
        assert len(truncated) == target_length
        
        # Test padding
        short_audio = np.random.randn(11025)  # 0.5 seconds
        padded = audio_preprocessor.pad_or_truncate(short_audio, target_length)
        assert len(padded) == target_length
        
        # Test exact length
        exact_audio = np.random.randn(target_length)
        result = audio_preprocessor.pad_or_truncate(exact_audio, target_length)
        assert len(result) == target_length
        np.testing.assert_array_equal(result, exact_audio)
    
    @patch('librosa.load')
    async def test_preprocess_for_traditional_models(self, mock_load, audio_preprocessor, mock_audio_data):
        """Test preprocessing for traditional ML models."""
        mock_load.return_value = (mock_audio_data, 22050)
        
        features = await audio_preprocessor.preprocess_for_traditional_models(
            Path("test.wav")
        )
        
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1  # Flattened features
        assert len(features) > 0
    
    @patch('librosa.load')
    async def test_preprocess_for_deep_learning_models(self, mock_load, audio_preprocessor, mock_audio_data):
        """Test preprocessing for deep learning models."""
        mock_load.return_value = (mock_audio_data, 22050)
        
        spectrogram = await audio_preprocessor.preprocess_for_deep_learning_models(
            Path("test.wav")
        )
        
        assert isinstance(spectrogram, np.ndarray)
        assert spectrogram.ndim == 2  # 2D spectrogram
        assert spectrogram.shape[0] == 128  # n_mels
    
    def test_get_feature_names(self, audio_preprocessor):
        """Test feature name generation."""
        # Test MFCC only
        names = audio_preprocessor.get_feature_names(include_derivatives=False)
        assert len(names) == 13 * 50  # 13 MFCC * ~50 time frames (approximate)
        
        # Test with derivatives
        names_with_derivatives = audio_preprocessor.get_feature_names(include_derivatives=True)
        assert len(names_with_derivatives) == 39 * 50  # 39 features * ~50 time frames