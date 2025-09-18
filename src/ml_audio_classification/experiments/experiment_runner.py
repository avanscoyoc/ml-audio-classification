"""Experiment orchestration system for running ML audio classification experiments."""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import json
from datetime import datetime

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score

from ..core.exceptions import ExperimentError
from ..core.logging import LoggerMixin
from ..core.reproducibility import set_global_seeds, create_fold_seed
from ..config import settings
from ..data import DatasetManager, AudioSample
from ..models import ModelFactory


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    model_name: str
    species: str
    training_size: int
    fold_results: List[float]  # ROC-AUC scores for each fold
    mean_auc: float
    std_auc: float
    confidence_interval: Tuple[float, float]
    training_time: float
    total_samples: int
    experiment_timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    models: List[str]
    species: List[str]
    training_sizes: List[int]
    cv_folds: int = 5
    feature_type: str = "mfcc"
    random_seed: int = 42


class ExperimentRunner(LoggerMixin):
    """Main experiment runner for multi-factor ML experiments."""
    
    def __init__(self) -> None:
        """Initialize experiment runner."""
        self.results: List[ExperimentResult] = []
        self.model_factory = ModelFactory()
        
        # Set global random seeds for reproducibility
        set_global_seeds(settings.experiment.seed)
    
    async def run_single_experiment(
        self,
        model_name: str,
        species: str,
        training_size: int,
        cv_folds: int = None
    ) -> ExperimentResult:
        """Run a single experiment with given parameters.
        
        Args:
            model_name: Name of the model to use
            species: Species to experiment with
            training_size: Number of training samples per class
            cv_folds: Number of cross-validation folds
            
        Returns:
            ExperimentResult object with results
            
        Raises:
            ExperimentError: If experiment fails
        """
        if cv_folds is None:
            cv_folds = settings.experiment.cv_folds
        
        start_time = datetime.now()
        
        try:
            self.logger.info(
                "Starting experiment",
                model_name=model_name,
                species=species,
                training_size=training_size,
                cv_folds=cv_folds
            )
            
            # Create dataset manager
            async with DatasetManager() as dataset_manager:
                # Create balanced dataset
                samples = await dataset_manager.create_balanced_dataset(
                    species=species,
                    training_size=training_size,
                    model_name=model_name,  # This determines if 5s data is used for Perch
                    download_files=True
                )
                
                # Validate samples were created successfully
                if not samples:
                    raise ExperimentError(f"No samples created for {species} with training size {training_size}")
                
                # All models are deep learning - validate audio data is loaded
                missing_audio = [s for s in samples if s.audio_data is None]
                if missing_audio:
                    self.logger.warning(
                        f"{len(missing_audio)} samples missing audio data for {model_name}"
                    )
                
                # Create cross-validation splits
                cv_splits = dataset_manager.create_cv_splits(samples, cv_folds)
                
                # Run cross-validation
                fold_results = []
                for fold_idx, (train_samples, val_samples) in enumerate(cv_splits):
                    fold_result = await self._run_single_fold(
                        model_name=model_name,
                        species=species,
                        train_samples=train_samples,
                        val_samples=val_samples,
                        fold_idx=fold_idx
                    )
                    fold_results.append(fold_result)
                    
                    self.logger.info(
                        "Completed fold",
                        fold_idx=fold_idx,
                        auc_score=fold_result,
                        model_name=model_name,
                        species=species
                    )
            
            # Calculate statistics
            mean_auc = np.mean(fold_results)
            std_auc = np.std(fold_results)
            
            # Calculate 95% confidence interval
            confidence_interval = self._calculate_confidence_interval(fold_results)
            
            # Calculate training time
            training_time = (datetime.now() - start_time).total_seconds()
            
            # Create result object
            result = ExperimentResult(
                model_name=model_name,
                species=species,
                training_size=training_size,
                fold_results=fold_results,
                mean_auc=mean_auc,
                std_auc=std_auc,
                confidence_interval=confidence_interval,
                training_time=training_time,
                total_samples=len(samples),
                experiment_timestamp=start_time.isoformat()
            )
            
            self.results.append(result)
            
            self.logger.info(
                "Completed experiment",
                model_name=model_name,
                species=species,
                training_size=training_size,
                mean_auc=mean_auc,
                std_auc=std_auc,
                training_time=training_time
            )
            
            return result
            
        except Exception as e:
            raise ExperimentError(
                f"Failed to run experiment {model_name}/{species}/{training_size}: {str(e)}"
            )
    
    async def _run_single_fold(
        self,
        model_name: str,
        species: str,
        train_samples: List[AudioSample],
        val_samples: List[AudioSample],
        fold_idx: int
    ) -> float:
        """Run a single cross-validation fold.
        
        Args:
            model_name: Name of the model to use
            species: Species name
            train_samples: Training samples
            val_samples: Validation samples
            fold_idx: Fold index for logging
            
        Returns:
            ROC-AUC score for this fold
        """
        try:
            # Set deterministic seed for this fold
            fold_seed = create_fold_seed(fold_idx)
            set_global_seeds(fold_seed)
            
            # Create model instance with validation
            try:
                model = self.model_factory.create_model(model_name)
                self.logger.debug(
                    "Created model instance",
                    model_name=model_name,
                    model_class=type(model).__name__
                )
            except Exception as e:
                raise ExperimentError(f"Failed to create model {model_name}: {str(e)}")
            
            # Set species for bioacoustics models
            if hasattr(model, 'set_species'):
                model.set_species(species)
            
            # Train model with error handling
            try:
                training_metrics = await model.train(train_samples, val_samples)
                self.logger.debug(
                    "Training completed",
                    model_name=model_name,
                    fold_idx=fold_idx,
                    metrics=training_metrics
                )
            except Exception as e:
                self.logger.error(
                    f"Training failed for {model_name} fold {fold_idx}: {str(e)}"
                )
                # Cleanup and re-raise
                model.cleanup()
                raise
            
            # Get predictions with error handling
            try:
                predictions = await model.predict_proba(val_samples)
                
                # Validate predictions shape
                if predictions.shape[0] != len(val_samples):
                    raise ExperimentError(
                        f"Prediction shape mismatch: got {predictions.shape[0]}, expected {len(val_samples)}"
                    )
                if predictions.shape[1] != 2:
                    raise ExperimentError(
                        f"Expected binary classification probabilities, got shape {predictions.shape}"
                    )
                    
            except Exception as e:
                self.logger.error(
                    f"Prediction failed for {model_name} fold {fold_idx}: {str(e)}"
                )
                model.cleanup()
                raise
            
            # Extract true labels
            true_labels = np.array([sample.label for sample in val_samples])
            
            # Calculate ROC-AUC score
            if len(np.unique(true_labels)) > 1:  # Check if we have both classes
                auc_score = roc_auc_score(true_labels, predictions[:, 1])
            else:
                # If only one class in validation set, use a default score
                auc_score = 0.5
                self.logger.warning(
                    "Only one class in validation set",
                    fold_idx=fold_idx,
                    model_name=model_name,
                    species=species
                )
            
            # Cleanup model
            model.cleanup()
            
            return float(auc_score)
            
        except Exception as e:
            raise ExperimentError(
                f"Failed to run fold {fold_idx} for {model_name}/{species}: {str(e)}"
            )
    
    def _calculate_confidence_interval(
        self,
        scores: List[float],
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Calculate confidence interval for scores.
        
        Args:
            scores: List of scores
            confidence: Confidence level (default 0.95 for 95% CI)
            
        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if len(scores) < 2:
            return (0.0, 1.0)
        
        # Use t-distribution for small samples
        try:
            from scipy import stats
            
            mean = np.mean(scores)
            std_err = stats.sem(scores)
            df = len(scores) - 1
            t_value = stats.t.ppf((1 + confidence) / 2, df)
            
            margin_error = t_value * std_err
            lower_bound = max(0.0, mean - margin_error)
            upper_bound = min(1.0, mean + margin_error)
            
            return (float(lower_bound), float(upper_bound))
            
        except ImportError:
            # Fallback to normal approximation if scipy not available
            self.logger.warning("scipy not available, using normal approximation for confidence intervals")
            mean = np.mean(scores)
            std = np.std(scores)
            z_value = 1.96  # 95% confidence interval
            margin_error = z_value * (std / np.sqrt(len(scores)))
            
            lower_bound = max(0.0, mean - margin_error)
            upper_bound = min(1.0, mean + margin_error)
            
            return (float(lower_bound), float(upper_bound))
    
    async def run_multi_factor_experiment(
        self,
        config: ExperimentConfig,
        max_concurrent: int = None
    ) -> List[ExperimentResult]:
        """Run multi-factor experiment across models, species, and training sizes.
        
        Args:
            config: Experiment configuration
            max_concurrent: Maximum number of concurrent experiments
            
        Returns:
            List of ExperimentResult objects
        """
        if max_concurrent is None:
            max_concurrent = settings.experiment.max_workers
        
        # Generate all experiment combinations
        experiments = []
        for model_name in config.models:
            for species in config.species:
                # Adjust training sizes based on available data
                adjusted_sizes = await self._get_adjusted_training_sizes(
                    species, config.training_sizes, model_name
                )
                
                for training_size in adjusted_sizes:
                    experiments.append((model_name, species, training_size))
        
        self.logger.info(
            "Starting multi-factor experiment",
            total_experiments=len(experiments),
            max_concurrent=max_concurrent,
            models=config.models,
            species=config.species
        )
        
        # Run experiments with limited concurrency
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def run_single_with_semaphore(model_name, species, training_size):
            async with semaphore:
                return await self.run_single_experiment(
                    model_name, species, training_size, config.cv_folds
                )
        
        # Execute all experiments
        tasks = [
            run_single_with_semaphore(model_name, species, training_size)
            for model_name, species, training_size in experiments
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and log them
        successful_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                experiment = experiments[i]
                self.logger.error(
                    "Experiment failed",
                    model_name=experiment[0],
                    species=experiment[1],
                    training_size=experiment[2],
                    error=str(result)
                )
            else:
                successful_results.append(result)
        
        self.logger.info(
            "Completed multi-factor experiment",
            total_experiments=len(experiments),
            successful_experiments=len(successful_results),
            failed_experiments=len(experiments) - len(successful_results)
        )
        
        return successful_results
    
    async def _get_adjusted_training_sizes(
        self,
        species: str,
        requested_sizes: List[int],
        model_name: str
    ) -> List[int]:
        """Get adjusted training sizes based on available data for a species.
        
        Args:
            species: Species name
            requested_sizes: Requested training sizes
            model_name: Model name (affects data source)
            
        Returns:
            List of adjusted training sizes that are feasible
        """
        try:
            # Determine if we should use 5-second data
            use_5s_data = model_name.lower() == "perch"
            
            # Get data availability
            async with DatasetManager() as dataset_manager:
                summary = await dataset_manager.get_species_data_summary(
                    species, use_5s_data
                )
            
            max_size = summary["max_usable_training_size"]
            
            # Filter and adjust sizes
            adjusted_sizes = [size for size in requested_sizes if size <= max_size]
            
            if not adjusted_sizes:
                # If no sizes are feasible, use the maximum available
                adjusted_sizes = [max_size] if max_size > 0 else []
            
            self.logger.debug(
                "Adjusted training sizes",
                species=species,
                model_name=model_name,
                use_5s_data=use_5s_data,
                max_available=max_size,
                requested_sizes=requested_sizes,
                adjusted_sizes=adjusted_sizes
            )
            
            return adjusted_sizes
            
        except Exception as e:
            self.logger.warning(
                "Failed to adjust training sizes, using original",
                species=species,
                model_name=model_name,
                error=str(e)
            )
            return requested_sizes
    
    def save_results(self, output_path: Path) -> None:
        """Save experiment results to JSON file.
        
        Args:
            output_path: Path to output file
        """
        try:
            # Convert results to serializable format
            results_data = {
                "experiment_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_experiments": len(self.results),
                    "settings": {
                        "seed": settings.experiment.seed,
                        "cv_folds": settings.experiment.cv_folds,
                        "max_workers": settings.experiment.max_workers,
                    }
                },
                "results": [result.to_dict() for result in self.results]
            }
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            self.logger.info(
                "Saved experiment results",
                output_path=str(output_path),
                total_results=len(self.results)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to save results",
                output_path=str(output_path),
                error=str(e)
            )
            raise ExperimentError(f"Failed to save results: {str(e)}")