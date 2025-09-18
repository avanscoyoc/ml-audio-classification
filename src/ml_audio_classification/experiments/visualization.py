"""Visualization and results output for ML experiments."""

import asyncio
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
from dataclasses import dataclass
import json
from datetime import datetime

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from matplotlib.figure import Figure
    import matplotlib.patches as mpatches
    PLOTTING_AVAILABLE = True
except ImportError:
    PLOTTING_AVAILABLE = False

from ..core.exceptions import VisualizationError
from ..core.logging import LoggerMixin
from ..config import settings
from ..data.gcs_client import GCSClient
from .experiment_runner import ExperimentResult


@dataclass
class PlotConfig:
    """Configuration for plotting."""
    figsize: Tuple[int, int] = (12, 8)
    dpi: int = 300
    style: str = "whitegrid"
    palette: str = "Set2"
    font_size: int = 12
    title_size: int = 14
    save_format: str = "png"


class ResultsVisualizer(LoggerMixin):
    """Visualizer for experiment results and ROC-AUC comparisons."""
    
    def __init__(self, plot_config: Optional[PlotConfig] = None) -> None:
        """Initialize results visualizer.
        
        Args:
            plot_config: Configuration for plotting
        """
        if not PLOTTING_AVAILABLE:
            self.logger.warning(
                "Plotting libraries not available. Install matplotlib, seaborn, pandas, numpy for visualization."
            )
        
        self.plot_config = plot_config or PlotConfig()
        self.gcs_client = GCSClient()
        
        # Set up plotting style if available
        if PLOTTING_AVAILABLE:
            plt.style.use('default')
            sns.set_style(self.plot_config.style)
            sns.set_palette(self.plot_config.palette)
            plt.rcParams.update({
                'font.size': self.plot_config.font_size,
                'axes.titlesize': self.plot_config.title_size,
                'axes.labelsize': self.plot_config.font_size,
                'xtick.labelsize': self.plot_config.font_size - 1,
                'ytick.labelsize': self.plot_config.font_size - 1,
                'legend.fontsize': self.plot_config.font_size - 1,
                'figure.dpi': self.plot_config.dpi
            })
    
    def _check_plotting_available(self) -> None:
        """Check if plotting libraries are available."""
        if not PLOTTING_AVAILABLE:
            raise VisualizationError(
                "Plotting libraries not available. "
                "Install matplotlib, seaborn, pandas, numpy for visualization."
            )
    
    def create_per_species_roc_charts(
        self,
        results: List[ExperimentResult],
        output_dir: Path,
        title_prefix: str = "Audio Classification"
    ) -> None:
        """Create per-species ROC-AUC charts as specified in CLAUDE.md.
        
        Creates charts with:
        - x-axis: Training sample size
        - y-axis: ROC-AUC score
        - Lines: Different models
        - Error bars: Confidence intervals from cross-validation
        
        Args:
            results: List of experiment results
            output_dir: Directory to save the charts
            title_prefix: Prefix for chart titles
        """
        self._check_plotting_available()
        
        # Group results by species
        species_results = {}
        for result in results:
            species = result.species
            if species not in species_results:
                species_results[species] = []
            species_results[species].append(result)
        
        # Create chart for each species
        for species, species_data in species_results.items():
            self._create_single_species_chart(
                species_data, 
                output_dir, 
                species, 
                title_prefix
            )
        
        self.logger.info(
            "Created per-species ROC-AUC charts",
            output_dir=str(output_dir),
            species_count=len(species_results)
        )
    
    def _create_single_species_chart(
        self,
        results: List[ExperimentResult],
        output_dir: Path,
        species: str,
        title_prefix: str
    ) -> None:
        """Create a single species ROC-AUC chart.
        
        Args:
            results: Results for this species
            output_dir: Output directory
            species: Species name
            title_prefix: Title prefix
        """
        # Prepare data for plotting
        model_data = {}
        
        for result in results:
            training_size = result.training_size
            model_name = result.model_name
            
            if model_name not in model_data:
                model_data[model_name] = {
                    'training_sizes': [],
                    'roc_aucs': [],
                    'confidence_intervals': []
                }
            
            # Calculate confidence interval from confidence_interval tuple
            # confidence_interval is (lower, upper), so CI width is half the range
            ci_lower, ci_upper = result.confidence_interval
            ci = (ci_upper - ci_lower) / 2
            
            model_data[model_name]['training_sizes'].append(training_size)
            model_data[model_name]['roc_aucs'].append(result.mean_auc)
            model_data[model_name]['confidence_intervals'].append(ci)
        
        # Create the plot
        plt.figure(figsize=self.plot_config.figsize)
        
        # Plot each model as a line with error bars
        for model_name, data in model_data.items():
            # Sort by training size for proper line plotting
            sorted_indices = np.argsort(data['training_sizes'])
            x_values = np.array(data['training_sizes'])[sorted_indices]
            y_values = np.array(data['roc_aucs'])[sorted_indices]
            errors = np.array(data['confidence_intervals'])[sorted_indices]
            
            plt.errorbar(
                x_values,
                y_values,
                yerr=errors,
                marker='o',
                label=model_name,
                capsize=4,
                capthick=1.5,
                linewidth=2.5,
                markersize=8,
                alpha=0.8
            )
        
        # Customize the plot according to CLAUDE.md specs
        plt.xlabel('Training Sample Size', fontsize=self.plot_config.font_size)
        plt.ylabel('ROC-AUC Score', fontsize=self.plot_config.font_size)
        plt.title(f'{title_prefix} - {species.title()} Classification Performance', 
                 fontsize=self.plot_config.title_size)
        
        plt.legend(
            frameon=True,
            fancybox=True,
            shadow=True,
            fontsize=self.plot_config.font_size - 1
        )
        
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.ylim(0.5, 1.0)  # ROC-AUC range
        
        # Set x-axis to start from 0 and use integer ticks
        x_min = 0
        x_max = max([max(data['training_sizes']) for data in model_data.values()])
        plt.xlim(x_min, x_max * 1.05)
        
        # Format axes
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().set_axisbelow(True)
        
        plt.tight_layout()
        
        # Save the plot
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"roc_auc_{species.lower().replace(' ', '_')}.{self.plot_config.save_format}"
        
        plt.savefig(
            output_path,
            format=self.plot_config.save_format,
            dpi=self.plot_config.dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        plt.close()
        
        self.logger.info(
            "Created species ROC-AUC chart",
            species=species,
            output_path=str(output_path)
        )
    
    def _plot_roc_by_training_size(self, df: pd.DataFrame, ax) -> None:
        """Plot ROC-AUC by training size."""
        # Group by model and create line plot
        for model in df['Model'].unique():
            model_data = df[df['Model'] == model]
            
            # Group by training size and calculate mean/std
            grouped = model_data.groupby('Training Size').agg({
                'ROC-AUC': ['mean', 'std'],
                'CV Std': 'mean'
            }).reset_index()
            
            training_sizes = grouped['Training Size']
            mean_auc = grouped[('ROC-AUC', 'mean')]
            std_auc = grouped[('ROC-AUC', 'std')].fillna(0)
            
            # Plot line with error bars
            ax.errorbar(
                training_sizes,
                mean_auc,
                yerr=std_auc,
                marker='o',
                label=model,
                capsize=3,
                capthick=1,
                linewidth=2,
                markersize=6
            )
        
        ax.set_xlabel('Training Size (samples)')
        ax.set_ylabel('ROC-AUC Score')
        ax.set_title('ROC-AUC vs Training Size by Model')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0.5, 1.0)
    
    def _plot_roc_by_model(self, df: pd.DataFrame, ax) -> None:
        """Plot ROC-AUC by model."""
        # Create box plot
        sns.boxplot(
            data=df,
            x='Model',
            y='ROC-AUC',
            hue='Training Size',
            ax=ax
        )
        
        ax.set_xlabel('Model')
        ax.set_ylabel('ROC-AUC Score')
        ax.set_title('ROC-AUC Distribution by Model')
        ax.tick_params(axis='x', rotation=45)
        ax.set_ylim(0.5, 1.0)
    
    def create_model_comparison_heatmap(
        self,
        results: List[ExperimentResult],
        output_path: Path,
        metric: str = "mean_auc"
    ) -> None:
        """Create heatmap comparing models across species and training sizes.
        
        Args:
            results: List of experiment results
            output_path: Path to save the heatmap
            metric: Metric to visualize (mean_auc, std_auc, training_time)
        """
        self._check_plotting_available()
        
        # Prepare data for heatmap
        heatmap_data = {}
        
        for result in results:
            model_name = result.model_name
            key = f"{result.species}_{result.training_size}"
            
            if key not in heatmap_data:
                heatmap_data[key] = {}
            
            value = getattr(result, metric)
            heatmap_data[key][model_name] = value
        
        # Convert to DataFrame
        df = pd.DataFrame(heatmap_data).T
        df.index.name = 'Species_TrainingSize'
        
        # Create heatmap
        plt.figure(figsize=self.plot_config.figsize)
        
        sns.heatmap(
            df,
            annot=True,
            cmap='RdYlBu_r',
            center=0.8,
            fmt='.3f',
            cbar_kws={'label': metric.replace('_', ' ').title()}
        )
        
        plt.title(f'{metric.replace("_", " ").title()} Comparison Across Models')
        plt.xlabel('Model')
        plt.ylabel('Species and Training Size')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path,
            format=self.plot_config.save_format,
            dpi=self.plot_config.dpi,
            bbox_inches='tight'
        )
        plt.close()
        
        self.logger.info(
            "Created model comparison heatmap",
            output_path=str(output_path),
            metric=metric
        )
    
    def create_training_progression_plot(
        self,
        results: List[ExperimentResult],
        output_path: Path,
        species: Optional[str] = None
    ) -> None:
        """Create plot showing model performance progression with training size.
        
        Args:
            results: List of experiment results
            output_path: Path to save the plot
            species: Filter results for specific species (optional)
        """
        self._check_plotting_available()
        
        # Filter by species if specified
        if species:
            results = [r for r in results if r.species == species]
        
        if not results:
            raise VisualizationError(f"No results found for species: {species}")
        
        # Prepare data
        plot_data = []
        for result in results:
            plot_data.append({
                'Model': result.model_name,
                'Training Size': result.training_size,
                'ROC-AUC': result.mean_auc,
                'Std Dev': result.std_auc,
                'Training Time': result.training_time,
                'Total Samples': result.total_samples
            })
        
        df = pd.DataFrame(plot_data)
        
        # Create subplots for different metrics
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        metrics = ['ROC-AUC', 'Std Dev', 'Training Time', 'Total Samples']
        
        for i, metric in enumerate(metrics):
            ax = axes[i // 2, i % 2]
            
            # Plot lines for each model
            for model in df['Model'].unique():
                model_data = df[df['Model'] == model].sort_values('Training Size')
                
                ax.plot(
                    model_data['Training Size'],
                    model_data[metric],
                    marker='o',
                    label=model,
                    linewidth=2,
                    markersize=6
                )
            
            ax.set_xlabel('Training Size')
            ax.set_ylabel(metric)
            ax.set_title(f'{metric} vs Training Size')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Set y-axis limits for performance metrics
            if metric in ['ROC-AUC']:
                ax.set_ylim(0.5, 1.0)
        
        # Set main title
        title = f"Training Progression - {species}" if species else "Training Progression - All Species"
        fig.suptitle(title, fontsize=self.plot_config.title_size + 2)
        
        plt.tight_layout()
        
        # Save plot
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(
            output_path,
            format=self.plot_config.save_format,
            dpi=self.plot_config.dpi,
            bbox_inches='tight'
        )
        plt.close()
        
        self.logger.info(
            "Created training progression plot",
            output_path=str(output_path),
            species=species or "all"
        )


class ResultsExporter(LoggerMixin):
    """Exporter for experiment results to various formats."""
    
    def __init__(self) -> None:
        """Initialize results exporter."""
        self.gcs_client = GCSClient()
    
    async def export_results_to_json(
        self,
        results: List[ExperimentResult],
        output_path: Path,
        include_metadata: bool = True
    ) -> None:
        """Export experiment results to JSON format.
        
        Args:
            results: List of experiment results
            output_path: Path to save the JSON file
            include_metadata: Whether to include metadata
        """
        export_data = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "total_experiments": len(results),
                "total_models": len(set(result.model_name for result in results)),
                "species": list(set(result.species for result in results)),
                "training_sizes": list(set(result.training_size for result in results))
            } if include_metadata else {},
            "results": [
                {
                    "model_name": result.model_name,
                    "species": result.species,
                    "training_size": result.training_size,
                    "fold_results": result.fold_results,
                    "mean_auc": result.mean_auc,
                    "std_auc": result.std_auc,
                    "confidence_interval": result.confidence_interval,
                    "training_time": result.training_time,
                    "total_samples": result.total_samples,
                    "experiment_timestamp": result.experiment_timestamp
                }
                for result in results
            ]
        }
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to file
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        self.logger.info(
            "Exported results to JSON",
            output_path=str(output_path),
            total_results=len(results)
        )
    
    async def export_results_to_csv(
        self,
        results: List[ExperimentResult],
        output_path: Path
    ) -> None:
        """Export experiment results to CSV format.
        
        Args:
            results: List of experiment results
            output_path: Path to save the CSV file
        """
        try:
            import pandas as pd
            
            # Flatten results for CSV
            csv_data = []
            for result in results:
                csv_data.append({
                    'model_name': result.model_name,
                    'species': result.species,
                    'training_size': result.training_size,
                    'mean_auc': result.mean_auc,
                    'std_auc': result.std_auc,
                    'confidence_interval_lower': result.confidence_interval[0],
                    'confidence_interval_upper': result.confidence_interval[1],
                    'training_time': result.training_time,
                    'total_samples': result.total_samples,
                    'experiment_timestamp': result.experiment_timestamp,
                    'fold_results': ','.join(map(str, result.fold_results))
                })
            
            df = pd.DataFrame(csv_data)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Write to CSV
            df.to_csv(output_path, index=False)
            
            self.logger.info(
                "Exported results to CSV",
                output_path=str(output_path),
                total_rows=len(csv_data)
            )
            
        except ImportError:
            self.logger.error("pandas not available for CSV export")
            raise VisualizationError("pandas required for CSV export")
    
    async def upload_results_to_gcs(
        self,
        results: List[ExperimentResult],
        gcs_path: str,
        format: str = "json"
    ) -> None:
        """Upload experiment results to Google Cloud Storage.
        
        Args:
            results: List of experiment results
            gcs_path: GCS path (gs://bucket/path)
            format: Export format (json or csv)
        """
        try:
            # Create temporary file
            from tempfile import NamedTemporaryFile
            
            with NamedTemporaryFile(mode='w', suffix=f'.{format}', delete=False) as tmp_file:
                tmp_path = Path(tmp_file.name)
                
                if format == "json":
                    await self.export_results_to_json(results, tmp_path)
                elif format == "csv":
                    await self.export_results_to_csv(results, tmp_path)
                else:
                    raise VisualizationError(f"Unsupported format: {format}")
                
                # Upload to GCS
                # Extract relative path from gs:// URL
                if gcs_path.startswith("gs://"):
                    # Parse gs://bucket/path to get relative path
                    parts = gcs_path[5:].split('/', 1)
                    if len(parts) > 1:
                        relative_path = parts[1]  # path after bucket name
                    else:
                        relative_path = f"results_{format}.{format}"
                else:
                    relative_path = gcs_path
                
                await self.gcs_client.upload_results(tmp_path, relative_path)
                
                # Clean up temporary file
                tmp_path.unlink()
            
            self.logger.info(
                "Uploaded results to GCS",
                gcs_path=gcs_path,
                format=format,
                total_results=len(results)
            )
            
        except Exception as e:
            self.logger.error(
                "Failed to upload results to GCS",
                gcs_path=gcs_path,
                error=str(e)
            )
            raise VisualizationError(f"Failed to upload results: {str(e)}")
    
    async def upload_plots_to_gcs(
        self,
        plots_dir: Path,
        gcs_base_path: str
    ) -> None:
        """Upload visualization plots to GCS.
        
        Args:
            plots_dir: Local directory containing plot files
            gcs_base_path: Base GCS path for plots
        """
        if not plots_dir.exists():
            self.logger.warning("Plots directory does not exist", plots_dir=str(plots_dir))
            return
        
        try:
            # Upload all PNG files in the plots directory
            for plot_file in plots_dir.glob("*.png"):
                relative_path = f"{gcs_base_path}/{plot_file.name}"
                await self.gcs_client.upload_results(plot_file, relative_path)
                
                self.logger.info(
                    "Uploaded plot to GCS",
                    local_path=str(plot_file),
                    gcs_path=relative_path
                )
                
        except Exception as e:
            self.logger.error(
                "Failed to upload plots to GCS",
                plots_dir=str(plots_dir),
                error=str(e)
            )
            raise VisualizationError(f"Failed to upload plots: {str(e)}")