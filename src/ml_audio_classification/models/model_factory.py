"""Model factory for creating different audio classification models."""

from typing import Dict, Type, Any
from .base_model import BaseAudioModel
from .deep_learning_models import VGGModel, MobileNetModel, ResNetModel
from .specialized_models import BirdNETModel, PerchModel
from ..core.exceptions import ModelError


class ModelFactory:
    """Factory for creating audio classification models."""
    
    # Registry of available models - exactly 5 models as specified in CLAUDE.md
    _models: Dict[str, Type[BaseAudioModel]] = {
        "birdnet": BirdNETModel,
        "perch": PerchModel,
        "vgg": VGGModel,        # VGG for audio classification
        "mobilenet": MobileNetModel,  # MobileNet for audio classification
        "resnet": ResNetModel,  # ResNet for audio classification
    }
    
    @classmethod
    def create_model(
        cls,
        model_name: str,
        **kwargs: Any
    ) -> BaseAudioModel:
        """Create a model instance.
        
        Args:
            model_name: Name of the model to create
            **kwargs: Model-specific configuration parameters
            
        Returns:
            Initialized model instance
            
        Raises:
            ModelError: If model name is not recognized
        """
        if model_name not in cls._models:
            available_models = list(cls._models.keys())
            raise ModelError(
                f"Unknown model '{model_name}'. "
                f"Available models: {available_models}"
            )
        
        model_class = cls._models[model_name]
        return model_class(**kwargs)
    
    @classmethod
    def get_available_models(cls) -> list[str]:
        """Get list of available model names.
        
        Returns:
            List of available model names
        """
        return list(cls._models.keys())
    
    @classmethod
    def register_model(
        cls,
        model_name: str,
        model_class: Type[BaseAudioModel]
    ) -> None:
        """Register a new model class.
        
        Args:
            model_name: Name to register the model under
            model_class: Model class to register
        """
        cls._models[model_name] = model_class
    
    @classmethod
    def get_model_config_defaults(cls, model_name: str) -> Dict[str, Any]:
        """Get default configuration for a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Dictionary of default configuration parameters
            
        Raises:
            ModelError: If model name is not recognized
        """
        defaults = {
            "birdnet": {
                "confidence_threshold": 0.5,
                "sample_rate": 22050,
                "use_neural_head": False,  # Use embedding + classifier approach
            },
            "perch": {
                "embedding_dim": 1280,
                "use_5s_data": True,
                "sample_rate": 22050,
                "use_neural_head": False,  # Use embedding + classifier approach
            },
            "vgg": {
                "input_shape": (224, 224, 3),
                "freeze_base": True,
                "sample_rate": 22050,
                "use_neural_head": True,  # Use neural classification head
            },
            "mobilenet": {
                "input_shape": (224, 224, 3),
                "freeze_base": True,
                "sample_rate": 22050,
                "use_neural_head": True,  # Use neural classification head
            },
            "resnet": {
                "input_shape": (224, 224, 3),
                "freeze_base": True,
                "sample_rate": 22050,
                "use_neural_head": True,  # Use neural classification head
            },
        }
        
        if model_name not in defaults:
            raise ModelError(f"No default config for model '{model_name}'")
        
        return defaults[model_name]