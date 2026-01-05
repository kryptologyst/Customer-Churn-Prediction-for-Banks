"""
Customer Churn Prediction Package

A comprehensive machine learning system for predicting customer churn in banking.
"""

__version__ = "1.0.0"
__author__ = "AI Research Team"
__email__ = "research@example.com"

from .data.generator import CustomerDataGenerator
from .features.engineering import FeatureEngineer
from .models.churn_predictor import ChurnPredictor
from .models.evaluation import ChurnEvaluator
from .utils.config import ProjectConfig, DataConfig, ModelConfig, EvaluationConfig
from .utils.reproducibility import set_seed, get_device, ensure_reproducibility

__all__ = [
    "CustomerDataGenerator",
    "FeatureEngineer", 
    "ChurnPredictor",
    "ChurnEvaluator",
    "ProjectConfig",
    "DataConfig",
    "ModelConfig",
    "EvaluationConfig",
    "set_seed",
    "get_device",
    "ensure_reproducibility",
]
