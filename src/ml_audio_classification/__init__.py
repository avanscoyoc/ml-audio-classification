"""ML Audio Classification package for running experiments on bird calls and other audio data.

This package provides a comprehensive framework for running machine learning experiments
on audio classification tasks with support for:

- Multiple ML models (traditional ML and deep learning)
- Experiment orchestration and scheduling
- Data pipeline with Google Cloud Storage integration
- Results visualization and export
- Kubernetes deployment for cloud-native execution

Key Components:
- experiments: Experiment runner, scheduler, and visualization
- models: ML model implementations and factory
- data: Audio preprocessing and dataset management
- config: Configuration management with Pydantic
"""

__version__ = "0.1.0"

# Import key components for easy access
from .config import settings
from .experiments import (
    ExperimentRunner,
    ExperimentConfig,
    ExperimentResult,
    ExperimentScheduler,
    ResultsVisualizer,
    ResultsExporter
)
from .models import ModelFactory

__all__ = [
    "settings",
    "ExperimentRunner",
    "ExperimentConfig", 
    "ExperimentResult",
    "ExperimentScheduler",
    "ResultsVisualizer",
    "ResultsExporter",
    "ModelFactory"
]
__author__ = "Audio ML Team"
__email__ = "team@example.com"

from .config import Settings
from .core.exceptions import MLAudioClassificationError

__all__ = ["Settings", "MLAudioClassificationError"]