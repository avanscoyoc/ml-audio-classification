"""Base model interface for audio classification."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

from ..data.dataset_manager import AudioSample
from ..core.logging import LoggerMixin


class BaseAudioModel(ABC, LoggerMixin):
    """Abstract base class for audio classification models."""
    
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize base model.
        
        Args:
            model_name: Name of the model
            **kwargs: Model-specific configuration
        """
        self.model_name = model_name
        self.config = kwargs
        self.is_trained = False
        self._model = None
    
    @abstractmethod
    async def train(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Train the model on given samples.
        
        Args:
            train_samples: Training samples with features
            validation_samples: Optional validation samples
            
        Returns:
            Training metrics and metadata
        """
        pass
    
    @abstractmethod
    async def predict(
        self,
        samples: List[AudioSample]
    ) -> np.ndarray:
        """Make predictions on samples.
        
        Args:
            samples: Samples to predict on
            
        Returns:
            Prediction probabilities (Nx2 array for binary classification)
        """
        pass
    
    @abstractmethod
    async def predict_proba(
        self,
        samples: List[AudioSample]
    ) -> np.ndarray:
        """Get prediction probabilities.
        
        Args:
            samples: Samples to predict on
            
        Returns:
            Prediction probabilities
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration.
        
        Returns:
            Model information dictionary
        """
        return {
            "model_name": self.model_name,
            "is_trained": self.is_trained,
            "config": self.config,
        }
    
    def cleanup(self) -> None:
        """Clean up model resources."""
        if self._model is not None:
            del self._model
            self._model = None
        self.is_trained = False
        
        self.logger.debug(
            "Cleaned up model",
            model_name=self.model_name
        )


class FeatureBasedModel(BaseAudioModel):
    """Base class for models that work with extracted features."""
    
    def _prepare_features(
        self,
        samples: List[AudioSample]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training/prediction.
        
        Args:
            samples: List of samples with features
            
        Returns:
            Tuple of (features, labels)
        """
        if not samples:
            raise ValueError("No samples provided")
        
        if samples[0].features is None:
            raise ValueError("Samples do not have extracted features")
        
        # Stack features
        X = np.vstack([sample.features.reshape(1, -1) for sample in samples])
        y = np.array([sample.label for sample in samples])
        
        self.logger.debug(
            "Prepared features",
            model_name=self.model_name,
            feature_shape=X.shape,
            label_shape=y.shape
        )
        
        return X, y


class DeepLearningModel(BaseAudioModel):
    """Base class for deep learning models."""
    
    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """Initialize deep learning model."""
        super().__init__(model_name, **kwargs)
        self.input_shape = kwargs.get("input_shape")
        self.num_classes = kwargs.get("num_classes", 2)
    
    def _prepare_data(
        self,
        samples: List[AudioSample]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for deep learning models.
        
        Args:
            samples: List of samples
            
        Returns:
            Tuple of (data, labels)
        """
        if not samples:
            raise ValueError("No samples provided")
        
        # For deep learning models, we might work with raw audio or spectrograms
        # This is a placeholder - specific models will implement their own preparation
        labels = np.array([sample.label for sample in samples])
        
        return None, labels