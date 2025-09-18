"""Configuration validation utilities."""

from typing import List, Dict, Any
import logging

from ..config import settings
from ..models import ModelFactory
from .exceptions import ValidationError

logger = logging.getLogger(__name__)


def validate_experiment_config() -> Dict[str, Any]:
    """Validate experiment configuration and return validation report.
    
    Returns:
        Dictionary with validation results and warnings
    """
    report = {
        "valid": True,
        "warnings": [],
        "errors": [],
        "model_status": {},
        "dependency_status": {}
    }
    
    # Check model availability
    model_factory = ModelFactory()
    available_models = model_factory.get_available_models()
    
    for model_name in settings.model.available_models:
        try:
            # Try to create each model to check for issues
            test_model = model_factory.create_model(model_name)
            report["model_status"][model_name] = "available"
            test_model.cleanup()
        except Exception as e:
            report["model_status"][model_name] = f"error: {str(e)}"
            report["warnings"].append(f"Model {model_name} may have issues: {str(e)}")
    
    # Check dependencies
    dependency_checks = [
        ("numpy", "import numpy"),
        ("sklearn", "from sklearn.ensemble import RandomForestClassifier"),
        ("tensorflow", "import tensorflow as tf"),
        ("torch", "import torch"),
        ("librosa", "import librosa"),
        ("scipy", "from scipy import stats"),
    ]
    
    for dep_name, import_stmt in dependency_checks:
        try:
            exec(import_stmt)
            report["dependency_status"][dep_name] = "available"
        except ImportError:
            report["dependency_status"][dep_name] = "missing"
            if dep_name in ["numpy", "sklearn"]:
                report["errors"].append(f"Critical dependency {dep_name} is missing")
                report["valid"] = False
            else:
                report["warnings"].append(f"Optional dependency {dep_name} is missing")
    
    # Check species configuration
    if not settings.species:
        report["errors"].append("No species configured for experiments")
        report["valid"] = False
    
    # Check training sizes
    if not settings.training_sizes:
        report["errors"].append("No training sizes configured for experiments")
        report["valid"] = False
    elif max(settings.training_sizes) > settings.model.max_training_size:
        report["warnings"].append(
            f"Some training sizes exceed max_training_size ({settings.model.max_training_size})"
        )
    
    # GCP configuration check
    try:
        settings.gcp.validate_for_runtime(testing_mode=settings.testing_mode)
    except ValueError as e:
        if not settings.testing_mode:
            report["errors"].append(f"GCP configuration error: {str(e)}")
            report["valid"] = False
        else:
            report["warnings"].append(f"GCP configuration warning (testing mode): {str(e)}")
    
    # Log summary
    if report["valid"]:
        logger.info("Configuration validation passed")
        if report["warnings"]:
            logger.info(f"Configuration warnings: {len(report['warnings'])}")
    else:
        logger.error(f"Configuration validation failed with {len(report['errors'])} errors")
    
    return report


def validate_model_requirements(model_name: str) -> bool:
    """Validate that a specific model can be used.
    
    Args:
        model_name: Name of the model to validate
        
    Returns:
        True if model can be used, False otherwise
    """
    try:
        model_factory = ModelFactory()
        model = model_factory.create_model(model_name)
        model.cleanup()
        return True
    except Exception as e:
        logger.warning(f"Model {model_name} validation failed: {str(e)}")
        return False