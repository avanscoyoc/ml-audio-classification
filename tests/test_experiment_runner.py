"""Test experiment runner functionality."""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime

from ml_audio_classification.experiments.experiment_runner import (
    ExperimentRunner, 
    ExperimentConfig, 
    ExperimentResult,
    ModelResult
)


class TestExperimentConfig:
    """Test experiment configuration."""
    
    def test_config_creation(self):
        """Test experiment config creation."""
        config = ExperimentConfig(
            models=["random_forest", "vgg"],
            species=["coyote", "bullfrog"],
            training_sizes=[100, 500],
            cv_folds=5
        )
        
        assert config.models == ["random_forest", "vgg"]
        assert config.species == ["coyote", "bullfrog"]
        assert config.training_sizes == [100, 500]
        assert config.cv_folds == 5
    
    def test_config_validation(self):
        """Test config validation."""
        # Test invalid CV folds
        with pytest.raises(ValueError):
            ExperimentConfig(
                models=["random_forest"],
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


class TestModelResult:
    """Test model result data structure."""
    
    def test_model_result_creation(self):
        """Test model result creation."""
        result = ModelResult(
            model_name="random_forest",
            roc_auc=0.85,
            accuracy=0.80,
            precision=0.75,
            recall=0.85,
            f1_score=0.80,
            training_time=30.0,
            cv_scores=[0.82, 0.84, 0.86, 0.88, 0.85],
            cv_mean=0.85,
            cv_std_dev=0.02
        )
        
        assert result.model_name == "random_forest"
        assert result.roc_auc == 0.85
        assert len(result.cv_scores) == 5
        assert result.cv_mean == 0.85


class TestExperimentResult:
    """Test experiment result data structure."""
    
    def test_experiment_result_creation(self):
        """Test experiment result creation."""
        model_result = ModelResult(
            model_name="random_forest",
            roc_auc=0.85,
            accuracy=0.80,
            precision=0.75,
            recall=0.85,
            f1_score=0.80,
            training_time=30.0,
            cv_scores=[0.82, 0.84, 0.86, 0.88, 0.85],
            cv_mean=0.85,
            cv_std_dev=0.02
        )
        
        result = ExperimentResult(
            experiment_id="test_exp_1",
            species="coyote",
            training_size=100,
            cv_folds=5,
            start_time=datetime.now(),
            model_results={"random_forest": model_result}
        )
        
        assert result.experiment_id == "test_exp_1"
        assert result.species == "coyote"
        assert result.training_size == 100
        assert "random_forest" in result.model_results
    
    def test_duration_calculation(self):
        """Test duration calculation."""
        start_time = datetime.now()
        result = ExperimentResult(
            experiment_id="test_exp_1",
            species="coyote",
            training_size=100,
            cv_folds=5,
            start_time=start_time,
            model_results={}
        )
        
        # Initially no end time, so duration should be None
        assert result.duration_minutes is None
        
        # Set end time
        end_time = start_time
        result.end_time = end_time
        
        # Duration should be 0 (or very close)
        assert result.duration_minutes is not None
        assert result.duration_minutes >= 0


class TestExperimentRunner:
    """Test experiment runner functionality."""
    
    @pytest.fixture
    def experiment_runner(self):
        """Create experiment runner instance."""
        return ExperimentRunner()
    
    @pytest.fixture
    def sample_config(self):
        """Create sample experiment configuration."""
        return ExperimentConfig(
            models=["random_forest"],
            species=["coyote"],
            training_sizes=[100],
            cv_folds=3  # Smaller for testing
        )
    
    def test_runner_initialization(self, experiment_runner):
        """Test runner initialization."""
        assert experiment_runner.dataset_manager is not None
        assert experiment_runner.model_factory is not None
    
    @patch('ml_audio_classification.experiments.experiment_runner.DatasetManager')
    @patch('ml_audio_classification.experiments.experiment_runner.ModelFactory')
    async def test_run_single_experiment(self, mock_model_factory, mock_dataset_manager, experiment_runner):
        """Test running a single experiment."""
        # Mock dataset manager
        mock_dataset = AsyncMock()
        mock_dataset.create_balanced_dataset.return_value = (
            [[1, 2, 3], [4, 5, 6]],  # X_train
            [0, 1],  # y_train
            [[7, 8, 9]],  # X_test
            [1]  # y_test
        )
        mock_dataset_manager.return_value.__aenter__.return_value = mock_dataset
        
        # Mock model
        mock_model = AsyncMock()
        mock_model.train.return_value = None
        mock_model.predict_proba.return_value = [[0.2, 0.8]]
        mock_model.predict.return_value = [1]
        mock_model_factory.create_model.return_value = mock_model
        
        # Mock cross-validation
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = [0.8, 0.85, 0.9]
            
            result = await experiment_runner.run_single_experiment(
                experiment_id="test_exp",
                model_name="random_forest",
                species="coyote",
                training_size=100,
                cv_folds=3
            )
        
        assert result.experiment_id == "test_exp"
        assert result.species == "coyote"
        assert result.training_size == 100
        assert "random_forest" in result.model_results
        
        model_result = result.model_results["random_forest"]
        assert model_result.model_name == "random_forest"
        assert model_result.roc_auc > 0
        assert len(model_result.cv_scores) == 3
    
    async def test_run_multi_factor_experiment(self, experiment_runner, sample_config):
        """Test running multi-factor experiment."""
        with patch.object(experiment_runner, 'run_single_experiment') as mock_single:
            # Mock single experiment results
            mock_result = ExperimentResult(
                experiment_id="test_exp_coyote_100",
                species="coyote",
                training_size=100,
                cv_folds=3,
                start_time=datetime.now(),
                model_results={}
            )
            mock_single.return_value = mock_result
            
            results = await experiment_runner.run_multi_factor_experiment(sample_config)
            
            # Should have 1 result (1 model * 1 species * 1 training size)
            assert len(results) == 1
            assert results[0].species == "coyote"
            assert results[0].training_size == 100
    
    def test_generate_experiment_id(self, experiment_runner):
        """Test experiment ID generation."""
        exp_id = experiment_runner._generate_experiment_id("coyote", 100, "random_forest")
        
        assert "coyote" in exp_id
        assert "100" in exp_id
        assert "random_forest" in exp_id
        assert len(exp_id) > 0