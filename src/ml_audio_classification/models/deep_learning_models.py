"""Deep learning models for audio classification."""

import asyncio
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import logging

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.applications import VGG11, MobileNetV2, ResNet50V2
    from tensorflow.keras.models import Model
    import librosa
except ImportError:
    tf = None
    keras = None
    layers = None
    VGG11 = None
    MobileNetV2 = None
    ResNet50V2 = None
    librosa = None

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    import torchvision.models as models
    import torchaudio
except ImportError:
    torch = None
    nn = None
    optim = None
    DataLoader = None
    TensorDataset = None
    models = None
    torchaudio = None

from .base_model import FeatureBasedModel
from ..data.dataset_manager import AudioSample
from ..core.exceptions import ModelError

logger = logging.getLogger(__name__)


class TransferLearningModel(FeatureBasedModel):
    """Base class for transfer learning with pre-trained models."""
    
    def __init__(self, 
                 model_name: str,
                 input_shape: Tuple[int, int, int] = (224, 224, 3),
                 freeze_base: bool = True,
                 **kwargs: Any) -> None:
        super().__init__(model_name, **kwargs)
        self.input_shape = input_shape
        self.freeze_base = freeze_base
        self.feature_extractor = None
        self.classification_head = None
        self.full_model = None
        self.built_model = False
        self.sample_rate = kwargs.get("sample_rate", 22050)
        self.use_neural_head = kwargs.get("use_neural_head", True)  # New parameter
        
    def _audio_to_spectrogram(self, audio_data: np.ndarray) -> np.ndarray:
        """Convert audio to mel-spectrogram and format for ImageNet models."""
        if librosa is None:
            raise ModelError("librosa not available for spectrogram conversion")
            
        # Create mel-spectrogram
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data,
            sr=self.sample_rate,
            n_mels=224,  # Match ImageNet input height
            fmax=8000,
            hop_length=512,
            n_fft=2048
        )
        
        # Convert to dB scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize to [0, 1]
        if mel_spec_db.max() > mel_spec_db.min():
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        else:
            mel_spec_norm = np.zeros_like(mel_spec_db)
        
        # Resize to match ImageNet input (224x224)
        if tf is not None:
            mel_spec_resized = tf.image.resize(
                mel_spec_norm[..., np.newaxis], 
                [224, 224]
            ).numpy()
        else:
            # Fallback: simple resize using numpy
            try:
                from scipy.ndimage import zoom
                zoom_factors = (224 / mel_spec_norm.shape[0], 224 / mel_spec_norm.shape[1])
                mel_spec_resized = zoom(mel_spec_norm, zoom_factors)
                mel_spec_resized = mel_spec_resized[..., np.newaxis]
            except ImportError:
                # Final fallback: basic interpolation
                self.logger.warning("scipy not available, using basic interpolation for spectrogram resizing")
                # Simple nearest-neighbor interpolation
                h_ratio = 224 / mel_spec_norm.shape[0]
                w_ratio = 224 / mel_spec_norm.shape[1]
                
                h_indices = np.round(np.arange(224) / h_ratio).astype(int)
                w_indices = np.round(np.arange(224) / w_ratio).astype(int)
                
                h_indices = np.clip(h_indices, 0, mel_spec_norm.shape[0] - 1)
                w_indices = np.clip(w_indices, 0, mel_spec_norm.shape[1] - 1)
                
                mel_spec_resized = mel_spec_norm[h_indices][:, w_indices]
                mel_spec_resized = mel_spec_resized[..., np.newaxis]
        
        # Convert to 3-channel (RGB-like) by repeating the channel
        mel_spec_rgb = np.repeat(mel_spec_resized, 3, axis=-1)
        
        return mel_spec_rgb
    
    async def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features using the pre-trained model."""
        if not self.built_model:
            self._build_feature_extractor()
        
        # Convert audio to spectrogram
        spectrogram = self._audio_to_spectrogram(audio_data)
        
        # Add batch dimension
        spectrogram_batch = np.expand_dims(spectrogram, axis=0)
        
        # Extract features using pre-trained model
        if self.feature_extractor is not None:
            features = self.feature_extractor.predict(spectrogram_batch, verbose=0)
            return features.flatten()
        else:
            # Fallback: return flattened spectrogram
            return spectrogram.flatten()[:1000]  # Limit size
    
    def _build_classification_head(self, feature_dim: int) -> None:
        """Build neural classification head for transfer learning."""
        if tf is None:
            raise ModelError("TensorFlow not available for neural classification head")
        
        # Create a simple but effective classification head
        self.classification_head = keras.Sequential([
            layers.Dense(128, activation='relu', name='fc1'),
            layers.Dropout(0.5, name='dropout1'),
            layers.Dense(64, activation='relu', name='fc2'),
            layers.Dropout(0.3, name='dropout2'),
            layers.Dense(2, activation='softmax', name='predictions')
        ], name=f'{self.model_name}_classification_head')
        
        # Build the head with proper input shape
        self.classification_head.build((None, feature_dim))
        
        # Compile the classification head
        self.classification_head.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _build_full_model(self) -> None:
        """Build full model combining feature extractor and classification head."""
        if tf is None or self.feature_extractor is None:
            return
        
        # Get feature dimension from feature extractor
        dummy_input = np.random.random((1,) + self.input_shape)
        dummy_features = self.feature_extractor.predict(dummy_input, verbose=0)
        feature_dim = dummy_features.shape[1]
        
        # Build classification head
        self._build_classification_head(feature_dim)
        
        # Create full model by connecting feature extractor and classification head
        inputs = self.feature_extractor.input
        features = self.feature_extractor.output
        predictions = self.classification_head(features)
        
        self.full_model = Model(inputs=inputs, outputs=predictions, name=f'{self.model_name}_full')
        
        # Compile full model
        self.full_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
    
    def _build_feature_extractor(self):
        """Build the feature extraction model - to be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _build_feature_extractor")


class VGGModel(TransferLearningModel):
    """VGG model using pre-trained ImageNet weights for transfer learning."""
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("vgg", **kwargs)
        if tf is None:
            raise ModelError("TensorFlow not available")
        self.logger.info("Initialized VGG transfer learning model")
        
    def _build_feature_extractor(self) -> None:
        """Build VGG11 feature extractor."""
        if self.feature_extractor is not None:
            return
            
        try:
            # Use VGG16 as base architecture (closest to VGG in Keras)
            base_model = keras.applications.VGG16(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Freeze base model layers for transfer learning
            if self.freeze_base:
                base_model.trainable = False
                self.logger.info(f"Frozen {len(base_model.layers)} layers in VGG16")
            
            # Add global average pooling to get fixed-size features
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            
            # Create feature extractor (no classification head)
            self.feature_extractor = Model(
                inputs=base_model.input,
                outputs=x,
                name="vgg_feature_extractor"
            )
            
            # Compile for efficiency
            self.feature_extractor.compile(optimizer='adam')
            self.built_model = True
            
            # Log model info
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.feature_extractor.trainable_weights])
            total_params = sum([tf.keras.backend.count_params(w) for w in self.feature_extractor.weights])
            
            self.logger.info(
                f"VGG - Total params: {total_params:,}, Trainable: {trainable_params:,}"
            )
            self.logger.info(
                f"Transfer learning ratio: {(total_params - trainable_params) / total_params * 100:.1f}% frozen"
            )
            
        except Exception as e:
            raise ModelError(f"Failed to build VGG feature extractor: {str(e)}")
    
    async def train(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Train using proper transfer learning with neural classification head."""
        try:
            # Build full model if not already built
            if not self.built_model:
                self._build_feature_extractor()
                if self.use_neural_head:
                    self._build_full_model()
            
            if self.use_neural_head and self.full_model is not None:
                # Try neural transfer learning approach first
                try:
                    return await self._train_neural_head(train_samples, validation_samples)
                except Exception as e:
                    self.logger.warning(
                        f"Neural training failed for {self.model_name}, falling back to feature extraction: {str(e)}"
                    )
                    # Fall back to feature extraction approach
                    self.use_neural_head = False
                    return await self._train_with_features(train_samples, validation_samples)
            else:
                # Use feature extraction approach
                return await self._train_with_features(train_samples, validation_samples)
            
        except Exception as e:
            raise ModelError(f"Failed to train {self.model_name}: {str(e)}")
    
    async def _train_neural_head(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Train with neural classification head."""
        # Prepare spectrograms and labels
        X_train = []
        y_train = []
        
        self.logger.info(f"Preparing spectrograms for {self.model_name} neural training...")
        
        for sample in train_samples:
            if sample.audio_data is not None:
                spectrogram = self._audio_to_spectrogram(sample.audio_data)
                X_train.append(spectrogram)
                y_train.append(sample.label)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Prepare validation data if available
        X_val, y_val = None, None
        if validation_samples:
            X_val = []
            y_val = []
            for sample in validation_samples:
                if sample.audio_data is not None:
                    spectrogram = self._audio_to_spectrogram(sample.audio_data)
                    X_val.append(spectrogram)
                    y_val.append(sample.label)
            X_val = np.array(X_val)
            y_val = np.array(y_val)
        
        # Train the full model
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self.full_model.fit(
            X_train, y_train,
            validation_data=validation_data,
            epochs=10,  # Can be configured
            batch_size=16,
            verbose=1
        )
        
        self.is_trained = True
        
        # Calculate final metrics
        train_pred = self.full_model.predict(X_train, verbose=0)
        train_proba = train_pred[:, 1]  # Probability of positive class
        
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, np.argmax(train_pred, axis=1))),
            "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
            "training_samples": len(train_samples),
            "final_train_loss": float(history.history['loss'][-1]),
            "model_type": "neural_transfer_learning"
        }
        
        if validation_data is not None:
            val_pred = self.full_model.predict(X_val, verbose=0)
            val_proba = val_pred[:, 1]
            
            metrics.update({
                "val_accuracy": float(accuracy_score(y_val, np.argmax(val_pred, axis=1))),
                "val_roc_auc": float(roc_auc_score(y_val, val_proba)),
                "final_val_loss": float(history.history['val_loss'][-1])
            })
        
        self.logger.info(
            f"Trained {self.model_name} with neural head",
            **metrics
        )
        
        return metrics
    
    async def _train_with_features(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Fallback training with feature extraction + RandomForest."""
        self.logger.info(f"Extracting {self.model_name} features for training...")
        
        # Extract features for all samples
        for sample in train_samples:
            if sample.features is None and sample.audio_data is not None:
                sample.features = await self._extract_features(sample.audio_data)
        
        if validation_samples:
            for sample in validation_samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
        
        # Use sklearn for final classification
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        # Prepare features and labels
        X_train, y_train = self._prepare_features(train_samples)
        
        # Train simple classifier on extracted features
        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Calculate metrics
        train_pred = self._model.predict(X_train)
        train_proba = self._model.predict_proba(X_train)[:, 1]
        
        metrics = {
            "train_accuracy": float(accuracy_score(y_train, train_pred)),
            "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
            "features_extracted": len(train_samples),
            "model_type": "feature_extraction_with_rf"
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
            f"Trained {self.model_name} with feature extraction",
            **metrics
        )
        
        return metrics
    
    async def predict_proba(self, samples: List[AudioSample]) -> np.ndarray:
        """Get prediction probabilities using the trained model."""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            if self.use_neural_head and self.full_model is not None:
                # Neural approach: predict directly with full model
                try:
                    spectrograms = []
                    for sample in samples:
                        if sample.audio_data is not None:
                            spectrogram = self._audio_to_spectrogram(sample.audio_data)
                            spectrograms.append(spectrogram)
                    
                    X = np.array(spectrograms)
                    probabilities = self.full_model.predict(X, verbose=0)
                    return probabilities
                except Exception as e:
                    self.logger.warning(f"Neural prediction failed, falling back to feature extraction: {str(e)}")
                    # Fall back to feature extraction
                    self.use_neural_head = False
            
            # Feature extraction approach (fallback or default)
            for sample in samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            X, _ = self._prepare_features(samples)
            return self._model.predict_proba(X)
                
        except Exception as e:
            raise ModelError(f"Failed to predict with {self.model_name}: {str(e)}")
    
    async def predict(self, samples: List[AudioSample]) -> np.ndarray:
        """Make predictions using the trained model."""
        probabilities = await self.predict_proba(samples)
        return np.argmax(probabilities, axis=1)


class MobileNetModel(TransferLearningModel):
    """MobileNet model using pre-trained ImageNet weights for transfer learning."""
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("mobilenet", **kwargs)
        if tf is None:
            raise ModelError("TensorFlow not available")
        self.logger.info("Initialized MobileNet transfer learning model")
        
    def _build_feature_extractor(self) -> None:
        """Build MobileNetV2 feature extractor."""
        if self.feature_extractor is not None:
            return
            
        try:
            # Use MobileNetV2 with ImageNet weights
            base_model = keras.applications.MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=self.input_shape
            )
            
            # Freeze base model layers for transfer learning
            if self.freeze_base:
                base_model.trainable = False
                self.logger.info(f"Frozen {len(base_model.layers)} layers in MobileNetV2")
            
            # Add global average pooling to get fixed-size features
            x = base_model.output
            x = layers.GlobalAveragePooling2D()(x)
            
            # Create feature extractor (no classification head)
            self.feature_extractor = Model(
                inputs=base_model.input,
                outputs=x,
                name="mobilenet_feature_extractor"
            )
            
            # Compile for efficiency
            self.feature_extractor.compile(optimizer='adam')
            self.built_model = True
            
            # Log model info
            trainable_params = sum([tf.keras.backend.count_params(w) for w in self.feature_extractor.trainable_weights])
            total_params = sum([tf.keras.backend.count_params(w) for w in self.feature_extractor.weights])
            
            self.logger.info(
                f"MobileNet - Total params: {total_params:,}, Trainable: {trainable_params:,}"
            )
            self.logger.info(
                f"Transfer learning ratio: {(total_params - trainable_params) / total_params * 100:.1f}% frozen"
            )
            
        except Exception as e:
            raise ModelError(f"Failed to build MobileNet feature extractor: {str(e)}")
    
    async def train(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Train using extracted MobileNetV2 features with a simple classifier."""
        try:
            # Extract features for all samples
            self.logger.info("Extracting MobileNetV2 features for training...")
            
            # Process samples to extract features
            for sample in train_samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            if validation_samples:
                for sample in validation_samples:
                    if sample.features is None and sample.audio_data is not None:
                        sample.features = await self._extract_features(sample.audio_data)
            
            # Use sklearn for final classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            # Prepare features and labels
            X_train, y_train = self._prepare_features(train_samples)
            
            # Train simple classifier on extracted features
            self._model = RandomForestClassifier(n_estimators=100, random_state=42)
            self._model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate metrics
            train_pred = self._model.predict(X_train)
            train_proba = self._model.predict_proba(X_train)[:, 1]
            
            metrics = {
                "train_accuracy": float(accuracy_score(y_train, train_pred)),
                "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
                "features_extracted": len(train_samples)
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
                "Trained MobileNetV2 transfer learning model",
                model_name=self.model_name,
                training_samples=len(train_samples),
                **metrics
            )
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"Failed to train MobileNetV2: {str(e)}")
    
    async def predict_proba(self, samples: List[AudioSample]) -> np.ndarray:
        """Get prediction probabilities using MobileNetV2 features."""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            # Extract features for prediction samples
            for sample in samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            # Prepare features
            X, _ = self._prepare_features(samples)
            
            # Get probabilities from trained classifier
            probabilities = self._model.predict_proba(X)
            
            return probabilities
            
        except Exception as e:
            raise ModelError(f"Failed to predict with MobileNetV2: {str(e)}")
    
    async def predict(self, samples: List[AudioSample]) -> np.ndarray:
        """Make predictions using MobileNetV2 features."""
        probabilities = await self.predict_proba(samples)
        return np.argmax(probabilities, axis=1)


class ResNetModel(TransferLearningModel):
    """ResNet model using pre-trained ImageNet weights for transfer learning."""
    
    def __init__(self, **kwargs: Any) -> None:
        super().__init__("resnet", **kwargs)
        if tf is None and torch is None:
            raise ModelError("Neither TensorFlow nor PyTorch available")
        self.use_torch = torch is not None
        self.logger.info("Initialized ResNet transfer learning model")
        
    def _build_feature_extractor(self) -> None:
        """Build ResNet18 feature extractor."""
        if self.feature_extractor is not None:
            return
            
        try:
            if self.use_torch:
                # Use PyTorch ResNet18
                self._build_torch_feature_extractor()
            else:
                # Fallback to TensorFlow ResNet50V2 (closest available)
                self._build_tf_feature_extractor()
                
        except Exception as e:
            raise ModelError(f"Failed to build ResNet18 feature extractor: {str(e)}")
    
    def _build_torch_feature_extractor(self) -> None:
        """Build PyTorch ResNet18 feature extractor."""
        # Load pre-trained ResNet18
        resnet18 = models.resnet18(pretrained=True)
        
        # Remove the final classification layer
        self.feature_extractor = nn.Sequential(*list(resnet18.children())[:-1])
        
        # Freeze parameters for transfer learning
        if self.freeze_base:
            for param in self.feature_extractor.parameters():
                param.requires_grad = False
            
        self.feature_extractor.eval()
        self.built_model = True
        
        # Log model info
        total_params = sum(p.numel() for p in self.feature_extractor.parameters())
        trainable_params = sum(p.numel() for p in self.feature_extractor.parameters() if p.requires_grad)
        
        self.logger.info(
            f"ResNet18 (PyTorch) - Total params: {total_params:,}, Trainable: {trainable_params:,}"
        )
        self.logger.info(
            f"Transfer learning ratio: {(total_params - trainable_params) / total_params * 100:.1f}% frozen"
        )
    
    def _build_tf_feature_extractor(self) -> None:
        """Build TensorFlow ResNet50V2 feature extractor as fallback."""
        # Use ResNet50V2 as closest equivalent to ResNet18 in Keras
        base_model = keras.applications.ResNet50V2(
            weights='imagenet',
            include_top=False,
            input_shape=self.input_shape
        )
        
        # Freeze base model layers for transfer learning
        if self.freeze_base:
            base_model.trainable = False
            self.logger.info(f"Frozen {len(base_model.layers)} layers in ResNet50V2")
        
        # Add global average pooling to get fixed-size features
        x = base_model.output
        x = layers.GlobalAveragePooling2D()(x)
        
        # Create feature extractor (no classification head)
        self.feature_extractor = Model(
            inputs=base_model.input,
            outputs=x,
            name="resnet18_feature_extractor"
        )
        
        # Compile for efficiency
        self.feature_extractor.compile(optimizer='adam')
        self.built_model = True
        
        # Log model info
        trainable_params = sum([tf.keras.backend.count_params(w) for w in self.feature_extractor.trainable_weights])
        total_params = sum([tf.keras.backend.count_params(w) for w in self.feature_extractor.weights])
        
        self.logger.info(
            f"ResNet50V2 (TensorFlow) - Total params: {total_params:,}, Trainable: {trainable_params:,}"
        )
        self.logger.info(
            f"Transfer learning ratio: {(total_params - trainable_params) / total_params * 100:.1f}% frozen"
        )
    
    async def _extract_features(self, audio_data: np.ndarray) -> np.ndarray:
        """Extract features using ResNet18."""
        if not self.built_model:
            self._build_feature_extractor()
        
        # Convert audio to spectrogram
        spectrogram = self._audio_to_spectrogram(audio_data)
        
        if self.use_torch:
            # PyTorch implementation
            import torch.nn.functional as F
            
            # Convert to tensor and add batch dimension
            spectrogram_tensor = torch.from_numpy(spectrogram).float()
            spectrogram_tensor = spectrogram_tensor.permute(2, 0, 1)  # HWC to CHW
            spectrogram_batch = spectrogram_tensor.unsqueeze(0)  # Add batch dimension
            
            # Extract features
            with torch.no_grad():
                features = self.feature_extractor(spectrogram_batch)
                features = F.adaptive_avg_pool2d(features, (1, 1))
                features = features.flatten()
                
            return features.numpy()
        else:
            # TensorFlow implementation
            spectrogram_batch = np.expand_dims(spectrogram, axis=0)
            features = self.feature_extractor.predict(spectrogram_batch, verbose=0)
            return features.flatten()
    
    async def train(
        self,
        train_samples: List[AudioSample],
        validation_samples: Optional[List[AudioSample]] = None
    ) -> Dict[str, Any]:
        """Train using extracted ResNet18 features with a simple classifier."""
        try:
            # Extract features for all samples
            self.logger.info("Extracting ResNet18 features for training...")
            
            # Process samples to extract features
            for sample in train_samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            if validation_samples:
                for sample in validation_samples:
                    if sample.features is None and sample.audio_data is not None:
                        sample.features = await self._extract_features(sample.audio_data)
            
            # Use sklearn for final classification
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.metrics import accuracy_score, roc_auc_score
            
            # Prepare features and labels
            X_train, y_train = self._prepare_features(train_samples)
            
            # Train simple classifier on extracted features
            self._model = RandomForestClassifier(n_estimators=100, random_state=42)
            self._model.fit(X_train, y_train)
            
            self.is_trained = True
            
            # Calculate metrics
            train_pred = self._model.predict(X_train)
            train_proba = self._model.predict_proba(X_train)[:, 1]
            
            metrics = {
                "train_accuracy": float(accuracy_score(y_train, train_pred)),
                "train_roc_auc": float(roc_auc_score(y_train, train_proba)),
                "features_extracted": len(train_samples)
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
                "Trained ResNet18 transfer learning model",
                model_name=self.model_name,
                training_samples=len(train_samples),
                **metrics
            )
            
            return metrics
            
        except Exception as e:
            raise ModelError(f"Failed to train ResNet18: {str(e)}")
    
    async def predict_proba(self, samples: List[AudioSample]) -> np.ndarray:
        """Get prediction probabilities using ResNet18 features."""
        if not self.is_trained:
            raise ModelError("Model is not trained")
        
        try:
            # Extract features for prediction samples
            for sample in samples:
                if sample.features is None and sample.audio_data is not None:
                    sample.features = await self._extract_features(sample.audio_data)
            
            # Prepare features
            X, _ = self._prepare_features(samples)
            
            # Get probabilities from trained classifier
            probabilities = self._model.predict_proba(X)
            
            return probabilities
            
        except Exception as e:
            raise ModelError(f"Failed to predict with ResNet18: {str(e)}")
    
    async def predict(self, samples: List[AudioSample]) -> np.ndarray:
        """Make predictions using ResNet18 features."""
        probabilities = await self.predict_proba(samples)
        return np.argmax(probabilities, axis=1)