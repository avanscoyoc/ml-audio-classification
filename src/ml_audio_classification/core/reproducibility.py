"""Reproducibility utilities for ensuring consistent random seeds across all components."""

import os
import random
from typing import Optional
import numpy as np

try:
    import tensorflow as tf
except ImportError:
    tf = None

try:
    import torch
except ImportError:
    torch = None

from ..config import settings
from .logging import LoggerMixin


class SeedManager(LoggerMixin):
    """Centralized seed management for reproducibility."""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize seed manager.
        
        Args:
            seed: Random seed to use. If None, uses config setting.
        """
        self.seed = seed if seed is not None else settings.experiment.seed
        self._is_seeded = False
    
    def set_global_seeds(self) -> None:
        """Set seeds for all random number generators."""
        if self._is_seeded:
            self.logger.debug(f"Seeds already set to {self.seed}")
            return
        
        # Python standard library
        random.seed(self.seed)
        
        # NumPy
        np.random.seed(self.seed)
        
        # Environment variable for Python hash seed
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        
        # TensorFlow
        if tf is not None:
            tf.random.set_seed(self.seed)
            
            # For TensorFlow 2.x deterministic behavior
            try:
                tf.config.experimental.enable_op_determinism()
            except AttributeError:
                # Fallback for older TensorFlow versions
                pass
        
        # PyTorch
        if torch is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
            
            # Make PyTorch deterministic
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        
        self._is_seeded = True
        
        self.logger.info(
            "Set global random seeds for reproducibility",
            seed=self.seed,
            tensorflow_available=tf is not None,
            pytorch_available=torch is not None
        )
    
    def get_seed(self) -> int:
        """Get the current seed value."""
        return self.seed
    
    def create_child_seed(self, offset: int = 1) -> int:
        """Create a child seed based on the main seed.
        
        Args:
            offset: Offset to add to the main seed
            
        Returns:
            Child seed value
        """
        return (self.seed + offset) % (2**32)


# Global seed manager instance
_global_seed_manager: Optional[SeedManager] = None


def get_seed_manager() -> SeedManager:
    """Get the global seed manager instance."""
    global _global_seed_manager
    if _global_seed_manager is None:
        _global_seed_manager = SeedManager()
        _global_seed_manager.set_global_seeds()
    return _global_seed_manager


def set_global_seeds(seed: Optional[int] = None) -> None:
    """Convenience function to set global seeds.
    
    Args:
        seed: Random seed to use. If None, uses config setting.
    """
    global _global_seed_manager
    _global_seed_manager = SeedManager(seed)
    _global_seed_manager.set_global_seeds()


def get_current_seed() -> int:
    """Get the current global seed value."""
    return get_seed_manager().get_seed()


def create_fold_seed(fold_idx: int) -> int:
    """Create a deterministic seed for a specific CV fold.
    
    Args:
        fold_idx: Cross-validation fold index
        
    Returns:
        Seed for this fold
    """
    return get_seed_manager().create_child_seed(1000 + fold_idx)