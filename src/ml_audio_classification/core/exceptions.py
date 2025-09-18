"""Core exceptions for the ML Audio Classification application."""


class MLAudioClassificationError(Exception):
    """Base exception for all ML Audio Classification errors."""
    
    def __init__(self, message: str, details: str = None) -> None:
        self.message = message
        self.details = details
        super().__init__(self.message)


class DataPipelineError(MLAudioClassificationError):
    """Exception raised during data pipeline operations."""
    pass


class AudioProcessingError(MLAudioClassificationError):
    """Exception raised during audio processing operations."""
    pass


class ModelError(MLAudioClassificationError):
    """Exception raised during model operations."""
    pass


class ExperimentError(MLAudioClassificationError):
    """Exception raised during experiment execution."""
    pass


class GCSError(MLAudioClassificationError):
    """Exception raised during Google Cloud Storage operations."""
    pass


class ConfigurationError(MLAudioClassificationError):
    """Exception raised for configuration-related errors."""
    pass


class ValidationError(MLAudioClassificationError):
    """Exception raised for data validation errors."""
    pass


class VisualizationError(MLAudioClassificationError):
    """Exception raised during visualization operations."""
    pass