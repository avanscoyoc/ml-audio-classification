"""Audio preprocessing utilities."""

import asyncio
from pathlib import Path
from typing import Tuple, Optional
import numpy as np

try:
    import librosa
    import soundfile as sf
except ImportError:
    librosa = None
    sf = None

from ..core.exceptions import DataPipelineError
from ..core.logging import LoggerMixin
from ..config import settings


class AudioPreprocessor(LoggerMixin):
    """Audio preprocessing for ML models."""
    
    def __init__(self) -> None:
        """Initialize audio preprocessor."""
        if librosa is None:
            raise DataPipelineError(
                "librosa not available. Install with: pip install librosa"
            )
        
        self.sample_rate = settings.audio.sample_rate
        self.duration = settings.audio.duration
        self.target_length = int(self.sample_rate * self.duration)
    
    async def load_and_preprocess(
        self,
        audio_path: Path,
        normalize: bool = True
    ) -> np.ndarray:
        """Load and preprocess an audio file.
        
        Args:
            audio_path: Path to audio file
            normalize: Whether to normalize audio
            
        Returns:
            Preprocessed audio array
            
        Raises:
            DataPipelineError: If preprocessing fails
        """
        try:
            # Load audio in executor to avoid blocking
            loop = asyncio.get_event_loop()
            
            # Define a wrapper function for librosa.load with proper arguments
            def load_audio():
                return librosa.load(str(audio_path), sr=self.sample_rate)
            
            audio, sr = await loop.run_in_executor(None, load_audio)
            
            # Ensure consistent length
            audio = self._ensure_length(audio)
            
            # Normalize if requested
            if normalize:
                audio = self._normalize_audio(audio)
            
            self.logger.debug(
                "Preprocessed audio",
                path=str(audio_path),
                shape=audio.shape,
                sample_rate=sr
            )
            
            return audio
            
        except Exception as e:
            raise DataPipelineError(
                f"Failed to preprocess audio {audio_path}: {str(e)}"
            )
    
    def _ensure_length(self, audio: np.ndarray) -> np.ndarray:
        """Ensure audio has target length.
        
        Args:
            audio: Input audio array
            
        Returns:
            Audio array with target length
        """
        if len(audio) > self.target_length:
            # Trim to target length
            return audio[:self.target_length]
        elif len(audio) < self.target_length:
            # Pad with zeros
            padding = self.target_length - len(audio)
            return np.pad(audio, (0, padding), mode='constant')
        else:
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """Normalize audio to [-1, 1] range.
        
        Args:
            audio: Input audio array
            
        Returns:
            Normalized audio array
        """
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val
        return audio
    
    async def extract_features(
        self,
        audio: np.ndarray,
        feature_type: str = "mfcc"
    ) -> np.ndarray:
        """Extract features from audio.
        
        Args:
            audio: Audio array
            feature_type: Type of features to extract
            
        Returns:
            Feature array
            
        Raises:
            DataPipelineError: If feature extraction fails
        """
        try:
            loop = asyncio.get_event_loop()
            
            if feature_type == "mfcc":
                def extract_mfcc():
                    return librosa.feature.mfcc(y=audio, sr=self.sample_rate, n_mfcc=13)
                features = await loop.run_in_executor(None, extract_mfcc)
            elif feature_type == "melspectrogram":
                def extract_melspec():
                    return librosa.feature.melspectrogram(y=audio, sr=self.sample_rate)
                features = await loop.run_in_executor(None, extract_melspec)
                # Convert to log scale
                features = librosa.power_to_db(features, ref=np.max)
            elif feature_type == "spectral_centroid":
                def extract_centroid():
                    return librosa.feature.spectral_centroid(y=audio, sr=self.sample_rate)
                features = await loop.run_in_executor(None, extract_centroid)
            else:
                raise DataPipelineError(f"Unknown feature type: {feature_type}")
            
            self.logger.debug(
                "Extracted features",
                feature_type=feature_type,
                shape=features.shape
            )
            
            return features
            
        except Exception as e:
            raise DataPipelineError(
                f"Failed to extract {feature_type} features: {str(e)}"
            )
    
    async def save_preprocessed(
        self,
        audio: np.ndarray,
        output_path: Path
    ) -> None:
        """Save preprocessed audio to file.
        
        Args:
            audio: Audio array to save
            output_path: Output file path
            
        Raises:
            DataPipelineError: If saving fails
        """
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                sf.write,
                str(output_path),
                audio,
                self.sample_rate
            )
            
            self.logger.debug(
                "Saved preprocessed audio",
                path=str(output_path),
                shape=audio.shape
            )
            
        except Exception as e:
            raise DataPipelineError(
                f"Failed to save audio to {output_path}: {str(e)}"
            )