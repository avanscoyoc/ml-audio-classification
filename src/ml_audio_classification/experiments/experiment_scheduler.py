"""Experiment scheduler for orchestrating complex ML experiment workflows."""

import asyncio
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timedelta

from ..core.exceptions import ExperimentError
from ..core.logging import LoggerMixin
from ..config import settings
from .experiment_runner import ExperimentRunner, ExperimentConfig, ExperimentResult


@dataclass
class ScheduledExperiment:
    """Container for a scheduled experiment."""
    experiment_id: str
    config: ExperimentConfig
    scheduled_time: datetime
    priority: int = 0
    max_retries: int = 3
    retry_count: int = 0
    status: str = "pending"  # pending, running, completed, failed
    result: Optional[ExperimentResult] = None
    error_message: Optional[str] = None


class ExperimentScheduler(LoggerMixin):
    """Scheduler for managing and executing ML experiments."""
    
    def __init__(self) -> None:
        """Initialize experiment scheduler."""
        self.experiments: Dict[str, ScheduledExperiment] = {}
        self.runner = ExperimentRunner()
        self.is_running = False
        self._stop_event = asyncio.Event()
        
        # Callbacks for experiment lifecycle events
        self.on_experiment_start: Optional[Callable] = None
        self.on_experiment_complete: Optional[Callable] = None
        self.on_experiment_failed: Optional[Callable] = None
    
    def schedule_experiment(
        self,
        experiment_id: str,
        config: ExperimentConfig,
        scheduled_time: Optional[datetime] = None,
        priority: int = 0
    ) -> str:
        """Schedule an experiment for execution.
        
        Args:
            experiment_id: Unique identifier for the experiment
            config: Experiment configuration
            scheduled_time: When to run the experiment (None for immediate)
            priority: Priority level (higher = more important)
            
        Returns:
            The experiment ID
        """
        if scheduled_time is None:
            scheduled_time = datetime.now()
        
        experiment = ScheduledExperiment(
            experiment_id=experiment_id,
            config=config,
            scheduled_time=scheduled_time,
            priority=priority
        )
        
        self.experiments[experiment_id] = experiment
        
        self.logger.info(
            "Scheduled experiment",
            experiment_id=experiment_id,
            scheduled_time=scheduled_time.isoformat(),
            priority=priority,
            models=config.models,
            species=config.species,
            training_sizes=config.training_sizes
        )
        
        return experiment_id
    
    def schedule_grid_search(
        self,
        base_experiment_id: str,
        models: List[str],
        species: List[str],
        training_sizes: List[int],
        cv_folds: int = 5,
        priority: int = 0
    ) -> List[str]:
        """Schedule a grid search across multiple parameters.
        
        Args:
            base_experiment_id: Base name for experiment IDs
            models: List of models to test
            species: List of species to test
            training_sizes: List of training sizes to test
            cv_folds: Number of CV folds
            priority: Priority level
            
        Returns:
            List of scheduled experiment IDs
        """
        experiment_ids = []
        
        for i, model in enumerate(models):
            for j, sp in enumerate(species):
                experiment_id = f"{base_experiment_id}_{model}_{sp}_{i}_{j}"
                
                config = ExperimentConfig(
                    models=[model],
                    species=[sp],
                    training_sizes=training_sizes,
                    cv_folds=cv_folds
                )
                
                # Stagger experiments slightly to avoid resource conflicts
                scheduled_time = datetime.now() + timedelta(seconds=i * 10 + j * 2)
                
                self.schedule_experiment(
                    experiment_id=experiment_id,
                    config=config,
                    scheduled_time=scheduled_time,
                    priority=priority
                )
                
                experiment_ids.append(experiment_id)
        
        self.logger.info(
            "Scheduled grid search",
            base_experiment_id=base_experiment_id,
            total_experiments=len(experiment_ids),
            models=models,
            species=species,
            training_sizes=training_sizes
        )
        
        return experiment_ids
    
    async def start_scheduler(
        self,
        check_interval: float = 30.0,
        max_concurrent: int = None
    ) -> None:
        """Start the experiment scheduler.
        
        Args:
            check_interval: How often to check for new experiments (seconds)
            max_concurrent: Maximum concurrent experiments
        """
        if self.is_running:
            self.logger.warning("Scheduler is already running")
            return
        
        if max_concurrent is None:
            max_concurrent = settings.experiment.max_workers
        
        self.is_running = True
        self._stop_event.clear()
        
        self.logger.info(
            "Starting experiment scheduler",
            check_interval=check_interval,
            max_concurrent=max_concurrent
        )
        
        # Create semaphore for limiting concurrent experiments
        semaphore = asyncio.Semaphore(max_concurrent)
        
        try:
            while self.is_running and not self._stop_event.is_set():
                # Get experiments ready to run
                ready_experiments = self._get_ready_experiments()
                
                if ready_experiments:
                    # Start experiments
                    tasks = []
                    for experiment in ready_experiments[:max_concurrent]:
                        task = asyncio.create_task(
                            self._run_experiment_with_semaphore(experiment, semaphore)
                        )
                        tasks.append(task)
                    
                    # Wait for at least one to complete or timeout
                    if tasks:
                        done, pending = await asyncio.wait(
                            tasks,
                            timeout=check_interval,
                            return_when=asyncio.FIRST_COMPLETED
                        )
                        
                        # Cancel pending tasks if stopping
                        if self._stop_event.is_set():
                            for task in pending:
                                task.cancel()
                else:
                    # No experiments ready, wait for check interval
                    try:
                        await asyncio.wait_for(
                            self._stop_event.wait(),
                            timeout=check_interval
                        )
                    except asyncio.TimeoutError:
                        pass  # Normal timeout, continue loop
                
        except Exception as e:
            self.logger.error(
                "Error in scheduler main loop",
                error=str(e)
            )
        finally:
            self.is_running = False
            self.logger.info("Experiment scheduler stopped")
    
    def _get_ready_experiments(self) -> List[ScheduledExperiment]:
        """Get experiments that are ready to run.
        
        Returns:
            List of experiments ready for execution, sorted by priority and time
        """
        now = datetime.now()
        ready = []
        
        for experiment in self.experiments.values():
            if (experiment.status == "pending" and 
                experiment.scheduled_time <= now):
                ready.append(experiment)
        
        # Sort by priority (descending) then by scheduled time (ascending)
        ready.sort(key=lambda x: (-x.priority, x.scheduled_time))
        
        return ready
    
    async def _run_experiment_with_semaphore(
        self,
        experiment: ScheduledExperiment,
        semaphore: asyncio.Semaphore
    ) -> None:
        """Run an experiment with concurrency control.
        
        Args:
            experiment: The experiment to run
            semaphore: Semaphore for concurrency control
        """
        async with semaphore:
            await self._run_single_experiment(experiment)
    
    async def _run_single_experiment(self, experiment: ScheduledExperiment) -> None:
        """Run a single scheduled experiment.
        
        Args:
            experiment: The experiment to run
        """
        experiment.status = "running"
        
        try:
            self.logger.info(
                "Starting scheduled experiment",
                experiment_id=experiment.experiment_id,
                retry_count=experiment.retry_count
            )
            
            # Call start callback if registered
            if self.on_experiment_start:
                await self._safe_callback(
                    self.on_experiment_start,
                    experiment
                )
            
            # Run the experiment
            results = await self.runner.run_multi_factor_experiment(
                experiment.config
            )
            
            # Mark as completed
            experiment.status = "completed"
            experiment.result = results
            
            # Call completion callback if registered
            if self.on_experiment_complete:
                await self._safe_callback(
                    self.on_experiment_complete,
                    experiment,
                    results
                )
            
            self.logger.info(
                "Completed scheduled experiment",
                experiment_id=experiment.experiment_id,
                total_results=len(results)
            )
            
        except Exception as e:
            experiment.retry_count += 1
            experiment.error_message = str(e)
            
            if experiment.retry_count <= experiment.max_retries:
                # Schedule for retry
                experiment.status = "pending"
                experiment.scheduled_time = datetime.now() + timedelta(
                    minutes=2 ** experiment.retry_count  # Exponential backoff
                )
                
                self.logger.warning(
                    "Experiment failed, scheduling retry",
                    experiment_id=experiment.experiment_id,
                    retry_count=experiment.retry_count,
                    max_retries=experiment.max_retries,
                    next_attempt=experiment.scheduled_time.isoformat(),
                    error=str(e)
                )
            else:
                # Max retries exceeded
                experiment.status = "failed"
                
                # Call failure callback if registered
                if self.on_experiment_failed:
                    await self._safe_callback(
                        self.on_experiment_failed,
                        experiment,
                        e
                    )
                
                self.logger.error(
                    "Experiment failed permanently",
                    experiment_id=experiment.experiment_id,
                    retry_count=experiment.retry_count,
                    error=str(e)
                )
    
    async def _safe_callback(self, callback: Callable, *args, **kwargs) -> None:
        """Safely execute a callback function.
        
        Args:
            callback: The callback function to execute
            *args: Positional arguments for the callback
            **kwargs: Keyword arguments for the callback
        """
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(*args, **kwargs)
            else:
                callback(*args, **kwargs)
        except Exception as e:
            self.logger.error(
                "Error in callback function",
                callback=callback.__name__,
                error=str(e)
            )
    
    async def stop_scheduler(self) -> None:
        """Stop the experiment scheduler gracefully."""
        self.logger.info("Stopping experiment scheduler")
        self.is_running = False
        self._stop_event.set()
    
    def get_experiment_status(self, experiment_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a scheduled experiment.
        
        Args:
            experiment_id: The experiment ID
            
        Returns:
            Dictionary with experiment status or None if not found
        """
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            return None
        
        return {
            "experiment_id": experiment.experiment_id,
            "status": experiment.status,
            "scheduled_time": experiment.scheduled_time.isoformat(),
            "priority": experiment.priority,
            "retry_count": experiment.retry_count,
            "max_retries": experiment.max_retries,
            "error_message": experiment.error_message,
            "has_results": experiment.result is not None
        }
    
    def get_all_experiment_status(self) -> List[Dict[str, Any]]:
        """Get status of all experiments.
        
        Returns:
            List of experiment status dictionaries
        """
        return [
            self.get_experiment_status(exp_id)
            for exp_id in self.experiments.keys()
        ]
    
    def save_experiment_state(self, output_path: Path) -> None:
        """Save the current state of all experiments.
        
        Args:
            output_path: Path to save the state file
        """
        try:
            state_data = {
                "scheduler_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "total_experiments": len(self.experiments),
                    "is_running": self.is_running
                },
                "experiments": [
                    self.get_experiment_status(exp_id)
                    for exp_id in self.experiments.keys()
                ]
            }
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            self.logger.info(
                "Saved experiment state",
                output_path=str(output_path),
                total_experiments=len(self.experiments)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to save experiment state",
                output_path=str(output_path),
                error=str(e)
            )
            raise ExperimentError(f"Failed to save experiment state: {str(e)}")