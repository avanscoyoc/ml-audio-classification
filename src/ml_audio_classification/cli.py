"""Main CLI interface for ML audio classification experiments."""

import asyncio
import argparse
from pathlib import Path
from typing import Optional, List
import sys
from datetime import datetime

from .config import settings
from .core.logging import LoggerMixin
from .core.reproducibility import set_global_seeds
from .core.validation import validate_experiment_config
from .experiments import (
    ExperimentRunner, 
    ExperimentConfig, 
    ExperimentScheduler,
    ResultsVisualizer,
    ResultsExporter
)


class MLAudioClassificationCLI(LoggerMixin):
    """Command-line interface for ML audio classification experiments."""
    
    def __init__(self) -> None:
        """Initialize CLI."""
        # Initialize global reproducibility settings first
        set_global_seeds(settings.experiment.seed)
        
        # Validate configuration
        validation_report = validate_experiment_config()
        if not validation_report["valid"]:
            self.logger.error("Configuration validation failed!")
            for error in validation_report["errors"]:
                self.logger.error(f"  - {error}")
            sys.exit(1)
        
        if validation_report["warnings"]:
            for warning in validation_report["warnings"]:
                self.logger.warning(f"  - {warning}")
        
        self.experiment_runner = ExperimentRunner()
        self.scheduler = ExperimentScheduler()
        self.visualizer = ResultsVisualizer()
        self.exporter = ResultsExporter()
    
    async def run_single_experiment(
        self,
        models: List[str],
        species: List[str],
        training_sizes: List[int],
        cv_folds: int = 5,
        seeds: List[int] = None,
        output_dir: Optional[Path] = None
    ) -> None:
        """Run a single experiment.
        
        Args:
            models: List of model names to test
            species: List of species to test
            training_sizes: List of training sizes to test
            cv_folds: Number of cross-validation folds
            seeds: List of random seeds for confidence intervals
            output_dir: Output directory for results
        """
        if seeds is None:
            seeds = [42]
            
        if output_dir is None:
            # Local staging directory - final results uploaded to GCS per CLAUDE.md: dse-staff/soundhub/results/
            output_dir = Path("./results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create experiment configuration
        config = ExperimentConfig(
            models=models,
            species=species,
            training_sizes=training_sizes,
            cv_folds=cv_folds
        )
        
        self.logger.info(
            "Starting single experiment",
            models=models,
            species=species,
            training_sizes=training_sizes,
            cv_folds=cv_folds,
            output_dir=str(output_dir)
        )
        
        try:
            # Run experiment
            results = await self.experiment_runner.run_multi_factor_experiment(config)
            
            if not results:
                self.logger.warning("No results generated from experiment")
                return
            
            # Test visualization in testing mode since we fixed the format mismatch
            # if settings.testing_mode:
            #     self.logger.info(
            #         "Testing mode: Skipping result exports",
            #         total_results=len(results),
            #         mean_auc=results[0].mean_auc if results else None
            #     )
            #     return
            
            # Export results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Export to JSON
            json_path = output_dir / f"results_{timestamp}.json"
            await self.exporter.export_results_to_json(results, json_path)
            
            # Export to CSV
            csv_path = output_dir / f"results_{timestamp}.csv"
            await self.exporter.export_results_to_csv(results, csv_path)
            
            # Create visualizations
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(exist_ok=True)
            
            try:
                # Per-species ROC-AUC charts (CLAUDE.md specification)
                self.visualizer.create_per_species_roc_charts(
                    results, 
                    plots_dir
                )
                
                # Optional: Overall model comparison heatmap
                heatmap_path = plots_dir / f"model_heatmap_{timestamp}.png"
                self.visualizer.create_model_comparison_heatmap(
                    results,
                    heatmap_path
                )
                
                self.logger.info("Created visualization plots", plots_dir=str(plots_dir))
                
            except Exception as e:
                self.logger.warning(
                    "Failed to create visualizations (plotting libraries may not be installed)",
                    error=str(e)
                )
            
            # Upload to GCS if configured and not disabled (CLAUDE.md specification: dse-staff/soundhub/results/)
            if settings.gcp.bucket_name and not settings.disable_gcs_upload:
                try:
                    gcs_path = f"gs://{settings.gcp.bucket_name}/{settings.gcp.results_path}/results_{timestamp}.json"
                    await self.exporter.upload_results_to_gcs(results, gcs_path, format="json")
                    
                    # Also upload plots if they exist
                    plots_dir = output_dir / "plots"
                    if plots_dir.exists():
                        await self.exporter.upload_plots_to_gcs(plots_dir, f"{settings.gcp.results_path}/plots_{timestamp}")
                    
                    self.logger.info("Uploaded results and plots to GCS", gcs_path=gcs_path)
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to upload results to GCS",
                        error=str(e)
                    )
            elif settings.disable_gcs_upload:
                self.logger.info("GCS upload disabled for local development")
            
            self.logger.info(
                "Experiment completed successfully",
                total_results=len(results),
                output_dir=str(output_dir)
            )
            
        except Exception as e:
            self.logger.error(
                "Experiment failed",
                error=str(e)
            )
            raise
    
    async def run_grid_search(
        self,
        models: List[str],
        species: List[str],
        training_sizes: List[int],
        cv_folds: int = 5,
        seeds: List[int] = None,
        output_dir: Optional[Path] = None,
        max_concurrent: int = 2
    ) -> None:
        """Run a grid search experiment using the scheduler.
        
        Args:
            models: List of model names to test
            species: List of species to test
            training_sizes: List of training sizes to test
            cv_folds: Number of cross-validation folds
            seeds: List of random seeds for confidence intervals
            output_dir: Output directory for results
            max_concurrent: Maximum concurrent experiments
        """
        if seeds is None:
            seeds = [42]
            
        if output_dir is None:
            # Local staging directory - final results uploaded to GCS per CLAUDE.md: dse-staff/soundhub/results/
            output_dir = Path("./grid_search_results")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(
            "Starting grid search experiment",
            models=models,
            species=species,
            training_sizes=training_sizes,
            cv_folds=cv_folds,
            max_concurrent=max_concurrent,
            output_dir=str(output_dir)
        )
        
        # Schedule grid search experiments
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_experiment_id = f"grid_search_{timestamp}"
        
        experiment_ids = self.scheduler.schedule_grid_search(
            base_experiment_id=base_experiment_id,
            models=models,
            species=species,
            training_sizes=training_sizes,
            cv_folds=cv_folds
        )
        
        self.logger.info(
            "Scheduled grid search experiments",
            total_experiments=len(experiment_ids),
            experiment_ids=experiment_ids[:5]  # Log first 5 IDs
        )
        
        # Set up result collection
        all_results = []
        
        def on_experiment_complete(experiment, results):
            """Callback for when an experiment completes."""
            all_results.extend(results)
            self.logger.info(
                "Grid search experiment completed",
                experiment_id=experiment.experiment_id,
                total_results_so_far=len(all_results)
            )
        
        self.scheduler.on_experiment_complete = on_experiment_complete
        
        try:
            # Start scheduler
            scheduler_task = asyncio.create_task(
                self.scheduler.start_scheduler(
                    check_interval=10.0,
                    max_concurrent=max_concurrent
                )
            )
            
            # Wait for all experiments to complete
            while True:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # Get status of all experiments
                statuses = self.scheduler.get_all_experiment_status()
                
                pending = len([s for s in statuses if s['status'] == 'pending'])
                running = len([s for s in statuses if s['status'] == 'running'])
                completed = len([s for s in statuses if s['status'] == 'completed'])
                failed = len([s for s in statuses if s['status'] == 'failed'])
                
                self.logger.info(
                    "Grid search progress",
                    pending=pending,
                    running=running,
                    completed=completed,
                    failed=failed,
                    total=len(statuses)
                )
                
                # Check if all experiments are done
                if pending == 0 and running == 0:
                    break
            
            # Stop scheduler
            await self.scheduler.stop_scheduler()
            scheduler_task.cancel()
            
            # Save scheduler state
            state_path = output_dir / f"scheduler_state_{timestamp}.json"
            self.scheduler.save_experiment_state(state_path)
            
            # Process and export results
            if all_results:
                # Export results
                json_path = output_dir / f"grid_search_results_{timestamp}.json"
                await self.exporter.export_results_to_json(all_results, json_path)
                
                csv_path = output_dir / f"grid_search_results_{timestamp}.csv"
                await self.exporter.export_results_to_csv(all_results, csv_path)
                
                # Create comprehensive visualizations
                plots_dir = output_dir / "plots"
                plots_dir.mkdir(exist_ok=True)
                
                try:
                    # Per-species ROC-AUC charts (CLAUDE.md specification)
                    self.visualizer.create_per_species_roc_charts(
                        all_results,
                        plots_dir
                    )
                    
                    # Optional: Overall model comparison heatmap
                    heatmap_path = plots_dir / f"grid_search_heatmap_{timestamp}.png"
                    self.visualizer.create_model_comparison_heatmap(
                        all_results,
                        heatmap_path
                    )
                    
                    self.logger.info("Created grid search visualizations")
                    
                except Exception as e:
                    self.logger.warning(
                        "Failed to create grid search visualizations",
                        error=str(e)
                    )
                
                self.logger.info(
                    "Grid search completed successfully",
                    total_results=len(all_results),
                    completed_experiments=completed,
                    failed_experiments=failed,
                    output_dir=str(output_dir)
                )
            else:
                self.logger.warning("No results collected from grid search")
                
        except KeyboardInterrupt:
            self.logger.info("Grid search interrupted by user")
            await self.scheduler.stop_scheduler()
            scheduler_task.cancel()
        except Exception as e:
            self.logger.error(
                "Grid search failed",
                error=str(e)
            )
            await self.scheduler.stop_scheduler()
            scheduler_task.cancel()
            raise


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description="ML Audio Classification Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single experiment with specific models and species
  python -m ml_audio_classification run-experiment \\
    --models birdnet vgg \\
    --species coyote \\
    --training-sizes 100 500 1000 \\
    --seeds 42 123 456
  
  # Run comprehensive grid search with multiple seeds for confidence intervals
  python -m ml_audio_classification grid-search \\
    --models birdnet perch vgg mobilenet resnet \\
    --species coyote bullfrog human_vocal \\
    --training-sizes 100 500 1000 2000 \\
    --seeds 42 123 456 789 \\
    --max-concurrent 3
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Single experiment command
    exp_parser = subparsers.add_parser(
        "run-experiment",
        help="Run a single multi-factor experiment"
    )
    exp_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["birdnet", "perch", "vgg", "mobilenet", "resnet"],
        help="Models to test"
    )
    exp_parser.add_argument(
        "--species",
        nargs="+",
        required=True,
        help="Species to test (coyote, bullfrog, human_vocal)"
    )
    exp_parser.add_argument(
        "--training-sizes",
        nargs="+",
        type=int,
        required=True,
        help="Training sizes to test"
    )
    exp_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    exp_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for confidence intervals (default: [42])"
    )
    exp_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local staging directory for results (final results uploaded to GCS: dse-staff/soundhub/results/)"
    )
    
    # Grid search command
    grid_parser = subparsers.add_parser(
        "grid-search",
        help="Run grid search across models and species"
    )
    grid_parser.add_argument(
        "--models",
        nargs="+",
        required=True,
        choices=["birdnet", "perch", "vgg", "mobilenet", "resnet"],
        help="Models to test"
    )
    grid_parser.add_argument(
        "--species",
        nargs="+",
        required=True,
        help="Species to test (coyote, bullfrog, human_vocal)"
    )
    grid_parser.add_argument(
        "--training-sizes",
        nargs="+",
        type=int,
        required=True,
        help="Training sizes to test"
    )
    grid_parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of cross-validation folds (default: 5)"
    )
    grid_parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42],
        help="Random seeds for confidence intervals (default: [42])"
    )
    grid_parser.add_argument(
        "--output-dir",
        type=Path,
        help="Local staging directory for results (final results uploaded to GCS: dse-staff/soundhub/results/)"
    )
    grid_parser.add_argument(
        "--max-concurrent",
        type=int,
        default=2,
        help="Maximum concurrent experiments (default: 2)"
    )
    
    return parser


async def main() -> None:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    cli = MLAudioClassificationCLI()
    
    try:
        if args.command == "run-experiment":
            await cli.run_single_experiment(
                models=args.models,
                species=args.species,
                training_sizes=args.training_sizes,
                cv_folds=args.cv_folds,
                seeds=args.seeds,
                output_dir=args.output_dir
            )
        elif args.command == "grid-search":
            await cli.run_grid_search(
                models=args.models,
                species=args.species,
                training_sizes=args.training_sizes,
                cv_folds=args.cv_folds,
                seeds=args.seeds,
                output_dir=args.output_dir,
                max_concurrent=args.max_concurrent
            )
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())