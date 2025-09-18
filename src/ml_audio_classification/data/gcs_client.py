"""Google Cloud Storage utilities for data access."""

import asyncio
from pathlib import Path
from typing import List, Optional, Tuple
import tempfile
import aiofiles

from google.cloud import storage
from google.cloud.exceptions import NotFound

from ..core.exceptions import GCSError
from ..core.logging import LoggerMixin
from ..config import settings

# Import for testing mode
try:
    import numpy as np
    import soundfile as sf
except ImportError:
    # These are only needed in testing mode
    np = None
    sf = None


class GCSClient(LoggerMixin):
    """Google Cloud Storage client for audio data access."""
    
    def __init__(self) -> None:
        """Initialize GCS client."""
        try:
            # Validate GCP configuration for runtime (unless in testing mode)
            settings.gcp.validate_for_runtime(testing_mode=settings.testing_mode)
            
            if settings.testing_mode:
                self.logger.info("Running in testing mode - GCS client will use mock data")
                self.client = None
                self.bucket = None
            else:
                self.client = storage.Client(project=settings.gcp.project_id)
                self.bucket = self.client.bucket(settings.gcp.bucket_name)
        except Exception as e:
            raise GCSError(f"Failed to initialize GCS client: {str(e)}")
    
    async def list_audio_files(
        self,
        species: str,
        label_type: str,
        use_5s_data: bool = False
    ) -> List[str]:
        """List audio files for a species and label type.
        
        Args:
            species: Species name (e.g., 'coyote', 'bullfrog')
            label_type: 'pos' or 'neg' for positive/negative samples
            use_5s_data: Whether to use 5-second data (for perch model)
            
        Returns:
            List of GCS blob names
            
        Raises:
            GCSError: If listing fails
        """
        try:
            data_suffix = "data_5s" if use_5s_data else "data"
            prefix = f"soundhub/data/audio/{species}/{data_suffix}/{label_type}/"
            
            self.logger.info(
                "Listing audio files",
                species=species,
                label_type=label_type,
                prefix=prefix,
                use_5s_data=use_5s_data
            )
            
            # Handle testing mode with mock data
            if settings.testing_mode:
                # Return mock audio file list for testing
                mock_files = [
                    f"{prefix}mock_audio_{i:03d}.wav" 
                    for i in range(100)  # Generate 100 mock files per category
                ]
                self.logger.info(f"Generated {len(mock_files)} mock audio files for {species}/{label_type}")
                return mock_files
            
            # Run in executor to avoid blocking
            loop = asyncio.get_event_loop()
            blobs = await loop.run_in_executor(
                None,
                lambda: list(self.bucket.list_blobs(prefix=prefix))
            )
            
            # Filter for audio files
            audio_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}
            audio_files = [
                blob.name for blob in blobs
                if Path(blob.name).suffix.lower() in audio_extensions
            ]
            
            self.logger.info(
                "Found audio files",
                species=species,
                label_type=label_type,
                count=len(audio_files)
            )
            
            return audio_files
            
        except Exception as e:
            raise GCSError(
                f"Failed to list audio files for {species}/{label_type}: {str(e)}"
            )
    
    async def download_audio_file(
        self,
        blob_name: str,
        local_path: Optional[Path] = None
    ) -> Path:
        """Download an audio file from GCS.
        
        Args:
            blob_name: GCS blob name
            local_path: Local path to save file, or None for temp file
            
        Returns:
            Path to downloaded file
            
        Raises:
            GCSError: If download fails
        """
        try:
            if local_path is None:
                # Create temporary file
                suffix = Path(blob_name).suffix
                temp_file = tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix
                )
                local_path = Path(temp_file.name)
                temp_file.close()
            
            # Handle testing mode with mock audio data
            if settings.testing_mode:
                # Generate mock audio file (simple sine wave)
                if np is None or sf is None:
                    raise GCSError("numpy and soundfile are required for testing mode")
                
                # Generate 5 seconds of mock audio (sine wave)
                sample_rate = settings.audio.sample_rate
                duration = 5.0
                t = np.linspace(0, duration, int(sample_rate * duration))
                # Mix different frequencies to simulate realistic audio
                audio_data = (
                    0.3 * np.sin(2 * np.pi * 440 * t) +  # A4 note
                    0.2 * np.sin(2 * np.pi * 880 * t) +  # A5 note
                    0.1 * np.random.normal(0, 0.1, len(t))  # Some noise
                )
                
                # Save mock audio file
                sf.write(str(local_path), audio_data, sample_rate)
                self.logger.info(f"Generated mock audio file: {local_path}")
                return local_path
            
            # Download blob
            blob = self.bucket.blob(blob_name)
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                blob.download_to_filename,
                str(local_path)
            )
            
            self.logger.debug(
                "Downloaded audio file",
                blob_name=blob_name,
                local_path=str(local_path)
            )
            
            return local_path
            
        except NotFound:
            raise GCSError(f"Audio file not found: {blob_name}")
        except Exception as e:
            raise GCSError(f"Failed to download {blob_name}: {str(e)}")
    
    async def upload_results(
        self,
        local_path: Path,
        gcs_path: str
    ) -> None:
        """Upload results file to GCS.
        
        Args:
            local_path: Local file path
            gcs_path: GCS destination path (relative to results bucket)
            
        Raises:
            GCSError: If upload fails
        """
        try:
            blob = self.bucket.blob(f"{settings.gcp.results_path}/{gcs_path}")
            
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                blob.upload_from_filename,
                str(local_path)
            )
            
            self.logger.info(
                "Uploaded results",
                local_path=str(local_path),
                gcs_path=gcs_path
            )
            
        except Exception as e:
            raise GCSError(f"Failed to upload results to {gcs_path}: {str(e)}")
    
    async def check_species_data_availability(
        self,
        species: str,
        use_5s_data: bool = False
    ) -> Tuple[int, int]:
        """Check available positive and negative samples for a species.
        
        Args:
            species: Species name
            use_5s_data: Whether to check 5-second data
            
        Returns:
            Tuple of (positive_count, negative_count)
            
        Raises:
            GCSError: If check fails
        """
        try:
            pos_files = await self.list_audio_files(species, "pos", use_5s_data)
            neg_files = await self.list_audio_files(species, "neg", use_5s_data)
            
            return len(pos_files), len(neg_files)
            
        except Exception as e:
            raise GCSError(
                f"Failed to check data availability for {species}: {str(e)}"
            )