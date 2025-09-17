"""Test configuration and settings."""

import pytest
from ml_audio_classification.config import Settings, GCPConfig, ExperimentConfig, ModelConfig, AudioConfig


class TestGCPConfig:
    """Test GCP configuration."""
    
    def test_default_values(self):
        """Test default configuration values."""
        config = GCPConfig(project_id="test-project")
        
        assert config.bucket_name == "dse-staff"
        assert config.audio_data_path == "soundhub/data/audio"
        assert config.audio_data_5s_path == "soundhub/data/audio"
        assert config.results_path == "soundhub/results"
    
    def test_required_project_id(self):
        """Test that project_id is required."""
        with pytest.raises(ValueError):
            GCPConfig()


class TestModelConfig:
    """Test model configuration."""
    
    def test_available_models(self):
        """Test available models list."""
        config = ModelConfig()
        
        expected_models = ["birdnet", "perch", "vgg", "mobilenet", "resnet"]
        assert config.available_models == expected_models
    
    def test_available_species(self):
        """Test available species list matches CLAUDE.md."""
        config = ModelConfig()
        
        expected_species = ["coyote", "bullfrog", "human_vocal"]
        assert config.available_species == expected_species
    
    def test_max_training_size(self):
        """Test max training size matches CLAUDE.md spec."""
        config = ModelConfig()
        
        assert config.max_training_size == 300


class TestExperimentConfig:
    """Test experiment configuration."""
    
    def test_default_cv_folds(self):
        """Test default cross-validation folds."""
        config = ExperimentConfig()
        
        assert config.cv_folds == 5
    
    def test_reproducible_seed(self):
        """Test reproducible random seed."""
        config = ExperimentConfig()
        
        assert config.seed == 42


class TestAudioConfig:
    """Test audio configuration."""
    
    def test_sample_rate(self):
        """Test audio sample rate."""
        config = AudioConfig()
        
        assert config.sample_rate == 22050


class TestSettings:
    """Test overall settings integration."""
    
    def test_settings_creation(self):
        """Test settings can be created with required GCP project ID."""
        # This would normally use environment variables
        # For testing, we'll create settings directly
        gcp_config = GCPConfig(project_id="test-project")
        experiment_config = ExperimentConfig()
        model_config = ModelConfig()
        audio_config = AudioConfig()
        
        settings = Settings(
            gcp=gcp_config,
            experiment=experiment_config,
            model=model_config,
            audio=audio_config
        )
        
        assert settings.gcp.project_id == "test-project"
        assert settings.experiment.seed == 42
        assert settings.model.max_training_size == 300
        assert settings.audio.sample_rate == 22050