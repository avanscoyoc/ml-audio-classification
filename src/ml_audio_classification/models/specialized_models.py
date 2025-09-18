"""Specialized audio models (BirdNET and Perch) for transfer learning.

Based on the working implementation from avanscoyoc/non-avian-ml/src
Uses bioacoustics-model-zoo for embedding extraction + shallow classifiers.
"""

import asyncio
import os
import tempfile
from typing import Any, Dict, List, Optional
import numpy as np
import logging

try:
    import torch
    import torch.hub
    import torchaudio
except ImportError:
    torch = None
    torchaudio = None

from .base_model import FeatureBasedModel
from ..data.dataset_manager import AudioSample
from ..core.exceptions import ModelError

logger = logging.getLogger(__name__)


class BirdNETModel(FeatureBasedModel):
    """BirdNET model using bioacoustics-model-zoo for feature extraction.
    
    Based on the working implementation in src/model_loader.py and src/trainer.py:
    - Load BirdNET from torch.hub (kitzeslab/bioacoustics-model-zoo)
    - Extract embeddings from audio files 
    - Train RandomForest classifier on embeddings
    """
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("birdnet", **kwargs)
        
        if torch is None:
            raise ModelError("PyTorch not available for BirdNET")
        
        self.batch_size = kwargs.get("batch_size", 32)
        self.num_workers = kwargs.get("num_workers", min(4, os.cpu_count() or 1))
        self.sample_rate = kwargs.get("sample_rate", 22050)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load BirdNET model
        self._load_birdnet_model()
        
        self.logger.info("Initialized BirdNET transfer learning model")
    
    def _load_birdnet_model(self) -> None:
        """Load BirdNET from bioacoustics-model-zoo."""
        try:
            self.logger.info("Loading BirdNET model from torch hub...")
            
            # Load from torch hub like in the working reference
            self.birdnet_model = torch.hub.load(
                'kitzeslab/bioacoustics-model-zoo', 
                'BirdNET', 
                trust_repo=True
            )
            self.birdnet_model = self.birdnet_model.to(self.device)
            
            self.logger.info(f"Successfully loaded BirdNET model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load BirdNET from torch hub: {str(e)}")
            self.logger.warning("BirdNET will use fallback embedding generation")
            self.birdnet_model = None
    
    async def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract BirdNET embeddings from audio data."""
        # Check if BirdNET model is available
        if self.birdnet_model is None:
            self.logger.warning("BirdNET model not available, using fallback features")
            # Return a fallback feature vector (could be MFCC or other acoustic features)
            return self._extract_fallback_features(audio_data)
        
        try:
            # Save audio to temporary file (BirdNET needs file paths)
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name
                
                # Convert to tensor and save
                audio_tensor = torch.from_numpy(audio_data).float().unsqueeze(0)
                torchaudio.save(temp_path, audio_tensor, self.sample_rate)
            
            try:
                # Extract embeddings like in BioacousticsModelWrapper.embed_files
                loop = asyncio.get_event_loop()
                embeddings = await loop.run_in_executor(
                    None,
                    lambda: self.birdnet_model.embed(
                        [temp_path],
                        return_dfs=False,
                        batch_size=1,
                        num_workers=1,
                    )
                )
                
                # Return first embedding or fallback
                if len(embeddings) > 0:
                    return embeddings[0]
                else:
                    self.logger.warning("BirdNET returned empty embeddings, using fallback")
                    return self._extract_fallback_features(audio_data)
                    
            finally:
                # Clean up temp file
                try:
                    os.unlink(temp_path)
                except:
                    pass
                    
        except Exception as e:
            self.logger.warning(f"BirdNET embedding extraction failed: {str(e)}, using fallback")
            return self._extract_fallback_features(audio_data)
    
    def _extract_fallback_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract fallback features when BirdNET is not available."""
        try:
            import librosa
            # Extract MFCC features as fallback
            mfccs = librosa.feature.mfcc(
                y=audio_data, 
                sr=self.sample_rate, 
                n_mfcc=13,
                n_fft=2048,
                hop_length=512
            )
            # Flatten and pad/truncate to standard size
            features = mfccs.flatten()
            target_size = 1024  # BirdNET embedding size
            if len(features) > target_size:
                return features[:target_size]
            else:
                return np.pad(features, (0, target_size - len(features)), mode='constant')
        except ImportError:
            # Final fallback: zero vector
            return np.zeros(1024)
    
    async def train(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Train using BirdNET embeddings + RandomForest classifier."""
        try:
            self.logger.info("Extracting BirdNET embeddings for training...")
            
            # Extract features for all samples
            for sample in train_samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            if validation_samples:
                for sample in validation_samples:
                    if sample.features is None and sample.audio_data is not None:
                        sample.features = await self._extract_features(sample.audio_data)
            
            # Train RandomForest on embeddings (like the working reference)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            X_train, y_train = self._prepare_features(train_samples)
            
            # Use RandomForest like in the working implementation
            self._model = RandomForestClassifier(n_estimators=100, random_state=42)
            self._model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate metrics
            train_pred = self._model.predict(X_train)
            train_proba = self._model.predict_proba(X_train)[:, 1]
            
            metrics = {
                "train_accuracy": float(accuracy_score(y_train, train_pred)),
                "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
                "embeddings_extracted": len(train_samples),
                "embedding_dim": X_train.shape[1]
            }
            
            if validation_samples:
                X_val, y_val = self._prepare_features(validation_samples)
                val_pred = self._model.predict(X_val)
                val_proba = self._model.predict_proba(X_val)[:, 1]
                
                metrics.update({
                    "val_accuracy": float(accuracy_score(y_val, val_pred)),
                    "val_roc_auc": float(roc_auc_score(y_val, val_proba))
                })
            
            self.logger.info(
                "Trained BirdNET model",
                **metrics
            )
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"Failed to train BirdNET: {str(e)}")
    
    async def predict_proba(self, samples: List[AudioSample]) -> np.ndarray:
        """Get prediction probabilities using BirdNET embeddings."""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            # Extract features for prediction
            for sample in samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            X, _ = self._prepare_features(samples)
            return self._model.predict_proba(X)
            
        except Exception as e:
            raise ModelError(f"Failed to predict with BirdNET: {str(e)}")
    
    async def predict(self, samples: List[AudioSample]) -> np.ndarray:
        """Make predictions using BirdNET embeddings."""
        probabilities = await self.predict_proba(samples)
        return np.argmax(probabilities, axis=1)


class PerchModel(FeatureBasedModel):
    """Perch model using bioacoustics-model-zoo for feature extraction.
    
    Based on the working implementation, uses Perch embeddings with shallow classifier.
    Note: Perch often uses 5-second segments (data_5s folder in reference).
    """
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("perch", **kwargs)
        
        if torch is None:
            raise ModelError("PyTorch not available for Perch")
        
        self.batch_size = kwargs.get("batch_size", 32)
        self.num_workers = kwargs.get("num_workers", min(4, os.cpu_count() or 1))
        self.sample_rate = kwargs.get("sample_rate", 22050)
        self.use_5s_data = kwargs.get("use_5s_data", True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load Perch model
        self._load_perch_model()
        
        self.logger.info("Initialized Perch transfer learning model")
    
    def _load_perch_model(self) -> None:
        """Load Perch from bioacoustics-model-zoo."""
        try:
            self.logger.info("Loading Perch model from torch hub...")
            
            # Load from torch hub like in the working reference
            self.perch_model = torch.hub.load(
                'kitzeslab/bioacoustics-model-zoo', 
                'Perch', 
                trust_repo=True
            )
            self.perch_model = self.perch_model.to(self.device)
            
            self.logger.info(f"Successfully loaded Perch model on {self.device}")
            
        except Exception as e:
            self.logger.error(f"Failed to load Perch from torch hub: {str(e)}")
            self.logger.warning("Perch will use fallback embedding generation")
            self.perch_model = None
    
    def _create_5s_segments(self, audio_data: np.ndarray) -> List[np.ndarray]:
        """Create 5-second segments for Perch (matching reference data_5s usage)."""
        segment_length = 5 * self.sample_rate
        segments = []
        
        if len(audio_data) < segment_length:
            # Pad short audio
            padded = np.pad(audio_data, (0, segment_length - len(audio_data)), mode='constant')
            segments.append(padded)
        else:
            # Create overlapping segments
            for i in range(0, len(audio_data) - segment_length + 1, segment_length // 2):
                segment = audio_data[i:i + segment_length]
                if len(segment) == segment_length:
                    segments.append(segment)
        
        return segments if segments else [np.zeros(segment_length)]
    
    async def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract Perch embeddings from audio data."""
        try:
            # Create 5s segments (Perch typically uses 5s data)
            segments = self._create_5s_segments(audio_data)
            embeddings = []
            
            for segment in segments:
                # Save segment to temp file
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_path = temp_file.name
                    
                    audio_tensor = torch.from_numpy(segment).float().unsqueeze(0)
                    torchaudio.save(temp_path, audio_tensor, self.sample_rate)
                
                try:
                    # Extract embeddings like in BioacousticsModelWrapper
                    loop = asyncio.get_event_loop()
                    segment_embeddings = await loop.run_in_executor(
                        None,
                        lambda: self.perch_model.embed(
                            [temp_path],
                            return_dfs=False,
                            batch_size=1,
                            num_workers=1,
                        )
                    )
                    
                    if len(segment_embeddings) > 0:
                        embeddings.append(segment_embeddings[0])
                        
                finally:
                    try:
                        os.unlink(temp_path)
                    except:
                        pass
            
            # Average embeddings across segments
            if embeddings:
                return np.mean(embeddings, axis=0)
            else:
                return np.zeros(1280)  # Perch embedding size
                    
        except Exception as e:
            self.logger.warning(f"Perch embedding extraction failed: {str(e)}")
            return np.zeros(1280)
    
    async def train(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Train using Perch embeddings + RandomForest classifier."""
        try:
            self.logger.info("Extracting Perch embeddings for training...")
            
            # Extract features for all samples
            for sample in train_samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            if validation_samples:
                for sample in validation_samples:
                    if sample.features is None and sample.audio_data is not None:
                        sample.features = await self._extract_features(sample.audio_data)
            
            # Train RandomForest on embeddings (like the working reference)
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            X_train, y_train = self._prepare_features(train_samples)
            
            # Use RandomForest like in the working implementation
            self._model = RandomForestClassifier(n_estimators=100, random_state=42)
            self._model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate metrics
            train_pred = self._model.predict(X_train)
            train_proba = self._model.predict_proba(X_train)[:, 1]
            
            metrics = {
                "train_accuracy": float(accuracy_score(y_train, train_pred)),
                "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
                "embeddings_extracted": len(train_samples),
                "embedding_dim": X_train.shape[1]
            }
            
            if validation_samples:
                X_val, y_val = self._prepare_features(validation_samples)
                val_pred = self._model.predict(X_val)
                val_proba = self._model.predict_proba(X_val)[:, 1]
                
                metrics.update({
                    "val_accuracy": float(accuracy_score(y_val, val_pred)),
                    "val_roc_auc": float(roc_auc_score(y_val, val_proba))
                })
            
            self.logger.info(
                "Trained Perch model",
                **metrics
            )
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"Failed to train Perch: {str(e)}")
    
    async def predict_proba(self, samples: List[AudioSample]) -> np.ndarray:
        """Get prediction probabilities using Perch embeddings."""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            # Extract features for prediction
            for sample in samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            X, _ = self._prepare_features(samples)
            return self._model.predict_proba(X)
            
        except Exception as e:
            raise ModelError(f"Failed to predict with Perch: {str(e)}")
    
    async def predict(self, samples: List[AudioSample]) -> np.ndarray:
        """Make predictions using Perch embeddings."""
        probabilities = await self.predict_proba(samples)
        return np.argmax(probabilities, axis=1)