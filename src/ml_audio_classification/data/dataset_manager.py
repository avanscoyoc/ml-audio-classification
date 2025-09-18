"""Dataset management for balanced sampling and cross-validation."""

import asyncio
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import tempfile
import shutil

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .gcs_client import GCSClient
from .audio_preprocessor import AudioPreprocessor
from ..core.exceptions import DataPipelineError, ValidationError
from ..core.logging import LoggerMixin
from ..core.reproducibility import get_current_seed
from ..config import settings


@dataclass
class AudioSample:
    """Represents an audio sample with metadata."""
    blob_name: str
    label: int  # 0 for negative, 1 for positive
    species: str
    local_path: Optional[Path] = None
    features: Optional[np.ndarray] = None
    audio_data: Optional[np.ndarray] = None


class DatasetManager(LoggerMixin):
    """Manages datasets for ML experiments with balanced sampling."""
    
    def __init__(self) -> None:
        """Initialize dataset manager."""
        self.gcs_client = GCSClient()
        self.audio_preprocessor = AudioPreprocessor()
        self.temp_dir = Path(tempfile.mkdtemp(prefix="ml_audio_"))
        
        # Seeds are managed globally by SeedManager - just log current seed
        current_seed = get_current_seed()
        self.logger.debug(f"DatasetManager using global seed: {current_seed}")
    
    async def __aenter__(self):
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup temp files."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    async def get_species_data_summary(
        self,
        species: str,
        use_5s_data: bool = False
    ) -> Dict[str, int]:
        """Get data availability summary for a species.
        
        Args:
            species: Species name
            use_5s_data: Whether to use 5-second data
            
        Returns:
            Dictionary with data counts
        """
        try:
            pos_count, neg_count = await self.gcs_client.check_species_data_availability(
                species, use_5s_data
            )
            
            max_balanced = min(pos_count, neg_count)
            
            summary = {
                "positive_samples": pos_count,
                "negative_samples": neg_count,
                "max_balanced_size": max_balanced,
                "max_usable_training_size": min(max_balanced, settings.model.max_training_size)
            }
            
            self.logger.info(
                "Species data summary",
                species=species,
                use_5s_data=use_5s_data,
                **summary
            )
            
            return summary
            
        except Exception as e:
            raise DataPipelineError(
                f"Failed to get data summary for {species}: {str(e)}"
            )
    
    async def create_balanced_dataset(
        self,
        species: str,
        training_size: int,
        model_name: str = "default",
        download_files: bool = True
    ) -> List[AudioSample]:
        """Create a balanced dataset for a species.
        
        Args:
            species: Species name
            training_size: Number of samples per class
            model_name: Model name (determines if 5s data should be used)
            download_files: Whether to download audio files locally
            
        Returns:
            List of AudioSample objects
            
        Raises:
            DataPipelineError: If dataset creation fails
            ValidationError: If insufficient data available
        """
        try:
            # Determine if we should use 5-second data based on model
            use_5s_data = model_name.lower() == "perch"
            
            # Check data availability
            summary = await self.get_species_data_summary(species, use_5s_data)
            
            if training_size > summary["max_usable_training_size"]:
                raise ValidationError(
                    f"Requested training size {training_size} exceeds maximum "
                    f"available balanced size {summary['max_usable_training_size']} "
                    f"for species {species} (use_5s_data={use_5s_data})"
                )
            
            # Get file lists
            pos_files = await self.gcs_client.list_audio_files(
                species, "pos", use_5s_data
            )
            neg_files = await self.gcs_client.list_audio_files(
                species, "neg", use_5s_data
            )
            
            # Sample balanced subsets
            pos_sample = random.sample(pos_files, training_size)
            neg_sample = random.sample(neg_files, training_size)
            
            # Create AudioSample objects
            samples = []
            
            # Positive samples
            for blob_name in pos_sample:
                samples.append(AudioSample(
                    blob_name=blob_name,
                    label=1,
                    species=species
                ))
            
            # Negative samples
            for blob_name in neg_sample:
                samples.append(AudioSample(
                    blob_name=blob_name,
                    label=0,
                    species=species
                ))
            
            # Shuffle the combined dataset
            random.shuffle(samples)
            
            # Download files if requested
            if download_files:
                await self._download_samples(samples)
            
            self.logger.info(
                "Created balanced dataset",
                species=species,
                training_size=training_size,
                total_samples=len(samples),
                downloaded=download_files,
                use_5s_data=use_5s_data,
                model_name=model_name
            )
            
            return samples
            
        except ValidationError:
            raise
        except Exception as e:
            raise DataPipelineError(
                f"Failed to create balanced dataset for {species}: {str(e)}"
            )
    
    async def _download_samples(self, samples: List[AudioSample]) -> None:
        """Download audio files for samples.
        
        Args:
            samples: List of AudioSample objects to download
        """
        # Create species subdirectories
        species_dirs = {}
        for sample in samples:
            if sample.species not in species_dirs:
                species_dir = self.temp_dir / sample.species
                species_dir.mkdir(exist_ok=True)
                species_dirs[sample.species] = species_dir
        
        # Download files concurrently (but limit concurrency)
        semaphore = asyncio.Semaphore(settings.experiment.max_workers)
        
        async def download_sample(sample: AudioSample) -> None:
            async with semaphore:
                # Generate local filename
                filename = Path(sample.blob_name).name
                local_path = species_dirs[sample.species] / filename
                
                # Download file
                await self.gcs_client.download_audio_file(
                    sample.blob_name,
                    local_path
                )
                sample.local_path = local_path
                
                # Load audio data for models that need raw audio
                try:
                    sample.audio_data = await self.audio_preprocessor.load_and_preprocess(
                        local_path, normalize=True
                    )
                    self.logger.debug(
                        "Loaded audio data",
                        blob_name=sample.blob_name,
                        audio_shape=sample.audio_data.shape
                    )
                except Exception as e:
                    self.logger.warning(
                        f"Failed to load audio data for {sample.blob_name}: {str(e)}"
                    )
                    sample.audio_data = None
        
        # Start downloads
        download_tasks = [download_sample(sample) for sample in samples]
        await asyncio.gather(*download_tasks)
        
        self.logger.info(
            "Downloaded samples",
            total_files=len(samples),
            temp_dir=str(self.temp_dir)
        )
    
    async def preprocess_samples(
        self,
        samples: List[AudioSample],
        feature_type: str = "mfcc"
    ) -> List[AudioSample]:
        """Preprocess audio samples and extract features.
        
        Args:
            samples: List of AudioSample objects
            feature_type: Type of features to extract
            
        Returns:
            List of preprocessed AudioSample objects
        """
        try:
            # Limit concurrency to avoid memory issues
            semaphore = asyncio.Semaphore(settings.experiment.max_workers)
            
            async def preprocess_sample(sample: AudioSample) -> AudioSample:
                async with semaphore:
                    if sample.local_path is None:
                        raise DataPipelineError(
                            f"No local path for sample {sample.blob_name}"
                        )
                    
                    # Load and preprocess audio
                    audio = await self.audio_preprocessor.load_and_preprocess(
                        sample.local_path
                    )
                    
                    # Extract features
                    features = await self.audio_preprocessor.extract_features(
                        audio, feature_type
                    )
                    
                    # Flatten features for sklearn compatibility
                    sample.features = features.flatten()
                    
                    return sample
            
            # Process samples
            tasks = [preprocess_sample(sample) for sample in samples]
            processed_samples = await asyncio.gather(*tasks)
            
            self.logger.info(
                "Preprocessed samples",
                total_samples=len(processed_samples),
                feature_type=feature_type,
                feature_shape=processed_samples[0].features.shape if processed_samples else None
            )
            
            return processed_samples
            
        except Exception as e:
            raise DataPipelineError(
                f"Failed to preprocess samples: {str(e)}"
            )
    
    def create_cv_splits(
        self,
        samples: List[AudioSample],
        n_folds: int = None
    ) -> List[Tuple[List[AudioSample], List[AudioSample]]]:
        """Create cross-validation splits.
        
        Args:
            samples: List of AudioSample objects
            n_folds: Number of folds (defaults to config value)
            
        Returns:
            List of (train, validation) sample tuples
        """
        if n_folds is None:
            n_folds = settings.experiment.cv_folds
        
        # Extract labels
        labels = [sample.label for sample in samples]
        
        # Create stratified splits
        skf = StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=get_current_seed()  # Use centrally managed seed
        )
        
        splits = []
        for train_idx, val_idx in skf.split(samples, labels):
            train_samples = [samples[i] for i in train_idx]
            val_samples = [samples[i] for i in val_idx]
            splits.append((train_samples, val_samples))
        
        self.logger.info(
            "Created CV splits",
            n_folds=n_folds,
            total_samples=len(samples),
            train_size=len(splits[0][0]) if splits else 0,
            val_size=len(splits[0][1]) if splits else 0
        )
        
        return splits