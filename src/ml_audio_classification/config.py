"""Application configuration using Pydantic Settings."""

from typing import List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class GCPConfig(BaseSettings):
    """Google Cloud Platform configuration."""
    
    project_id: Optional[str] = Field(default=None, description="GCP Project ID", alias="GCP_PROJECT_ID")
    bucket_name: str = Field(default="dse-staff", description="GCS bucket name", alias="GCS_BUCKET_NAME")
    credentials_path: Optional[str] = Field(default=None, description="Path to service account JSON")
    
    # Data paths matching CLAUDE.md specifications
    audio_data_path: str = Field(default="soundhub/data/audio", description="Audio data path in GCS")
    audio_data_5s_path: str = Field(default="soundhub/data/audio", description="5-second audio data path for Perch")
    results_path: str = Field(default="soundhub/results", description="Results output path in GCS")
    
    def validate_for_runtime(self, testing_mode: bool = False):
        """Validate that required fields are set for runtime operations."""
        if not testing_mode and not self.project_id:
            raise ValueError("GCP project_id is required for runtime operations. Set GCP_PROJECT_ID environment variable.")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


class ExperimentConfig(BaseModel):
    """Experiment configuration."""
    
    seed: int = Field(default=42, description="Random seed for reproducibility")
    max_workers: int = Field(default=4, description="Maximum number of worker processes")
    cv_folds: int = Field(default=5, description="Number of cross-validation folds")
    timeout_seconds: int = Field(default=3600, description="Experiment timeout in seconds")


class ModelConfig(BaseModel):
    """Model configuration."""
    
    available_models: List[str] = Field(
        default=["birdnet", "perch", "vgg", "mobilenet", "resnet"],
        description="Available model types"
    )
    available_species: List[str] = Field(
        default=["coyote", "bullfrog", "human_vocal"],
        description="Available species for classification (extensible)"
    )
    batch_size: int = Field(default=32, description="Training batch size")
    max_training_size: int = Field(default=300, description="Maximum training samples per class")


class AudioConfig(BaseModel):
    """Audio processing configuration."""
    
    sample_rate: int = Field(default=22050, description="Audio sample rate")
    duration: float = Field(default=5.0, description="Audio clip duration in seconds")


class Settings(BaseSettings):
    """Main application settings."""
    
    # Application
    log_level: str = Field(default="INFO", description="Logging level")
    debug: bool = Field(default=False, description="Debug mode")
    testing_mode: bool = Field(default=False, description="Testing mode - uses mock data instead of GCS")
    disable_gcs_upload: bool = Field(default=False, description="Disable GCS uploads (local development)")
    
    # GCP
    gcp: GCPConfig = Field(default_factory=GCPConfig)
    
    # Experiment
    experiment: ExperimentConfig = Field(default_factory=ExperimentConfig)
    
    # Models
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    # Audio
    audio: AudioConfig = Field(default_factory=AudioConfig)
    
    # Species and training sizes
    species: List[str] = Field(
        default=["coyote", "bullfrog", "human_vocal"],
        description="Species to experiment with"
    )
    training_sizes: List[int] = Field(
        default=[10, 25, 50, 100, 150, 200, 250, 300],
        description="Training set sizes to experiment with"
    )
    
    # Paths
    results_dir: str = Field(default="./results", description="Local results directory")
    
    # Health checks
    health_check_interval: int = Field(default=30, description="Health check interval in seconds")
    
    class Config:
        env_file = ".env"
        env_nested_delimiter = "__"
        case_sensitive = False


# Global settings instance
settings = Settings()