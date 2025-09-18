"""Test experiment runner functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ml_audio_classification.experiments.experiment_runner import (
    ExperimentRunner, 
    ExperimentConfig, 
    ExperimentResult
)


class TestExperimentConfig:
    """Test experiment configuration."""
    
    def test_config_creation(self):
        """Test experiment config creation."""
        config = ExperimentConfig(
            models=["birdnet", "vgg"],
            species=["coyote", "bullfrog"],
            training_sizes=[100, 500],
            cv_folds=5
        )
        
        assert config.models == ["birdnet", "vgg"]
        assert config.species == ["coyote", "bullfrog"]
        assert config.training_sizes == [100, 500]
        assert config.cv_folds == 5
    
    def test_config_validation(self):
        """Test config validation."""
        # Test invalid CV folds
        with pytest.raises(ValueError):
            ExperimentConfig(
                models=["birdnet"],
                species=["coyote"],
                training_sizes=[100],
                cv_folds=1  # Too few folds
            )
        
        # Test empty models
        with pytest.raises(ValueError):
            ExperimentConfig(
                models=[],
                species=["coyote"],
                training_sizes=[100],
                cv_folds=5
            )


class TestExperimentResult:
    """Test experiment result data structure."""
    
    def test_experiment_result_creation(self):
        """Test experiment result creation."""
        result = ExperimentResult(
            model_name="birdnet",
            species="coyote", 
            training_size=100,
            fold_results=[0.82, 0.84, 0.86, 0.88, 0.85],
            mean_auc=0.85,
            std_auc=0.02,
            confidence_interval=(0.83, 0.87)
        )
        
        assert result.model_name == "birdnet"
        assert result.species == "coyote"
        assert len(result.fold_results) == 5
        assert result.mean_auc == 0.85


class TestExperimentRunner:
    """Test experiment runner functionality."""
    
    @pytest.fixture
    def experiment_runner(self):
        """Create experiment runner instance."""
        return ExperimentRunner()
    
    def test_init(self, experiment_runner):
        """Test experiment runner initialization."""
        assert experiment_runner is not None
