"""
Configuration management for Customer Churn Prediction project.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
import yaml
from pathlib import Path


@dataclass
class DataConfig:
    """Configuration for data processing."""
    
    # Data paths
    raw_data_path: str = "data/raw"
    processed_data_path: str = "data/processed"
    features_path: str = "data/features"
    
    # Data parameters
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    
    # Time-based splits
    time_column: str = "date"
    train_start_date: Optional[str] = None
    train_end_date: Optional[str] = None
    test_start_date: Optional[str] = None
    test_end_date: Optional[str] = None
    
    # Feature engineering
    lookback_window: int = 30  # days
    feature_window: int = 90   # days
    label_window: int = 30     # days


@dataclass
class ModelConfig:
    """Configuration for model training."""
    
    # Model parameters
    model_type: str = "xgboost"  # xgboost, lightgbm, logistic_regression, ensemble
    n_estimators: int = 100
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    
    # Training parameters
    early_stopping_rounds: int = 10
    eval_metric: str = "auc"
    random_state: int = 42
    
    # Cross-validation
    cv_folds: int = 5
    cv_strategy: str = "time_series"  # time_series, stratified_kfold


@dataclass
class EvaluationConfig:
    """Configuration for model evaluation."""
    
    # Metrics
    primary_metric: str = "auc"
    secondary_metrics: list = None
    
    # Threshold optimization
    optimize_threshold: bool = True
    threshold_method: str = "youden"  # youden, f1, precision_recall
    
    # Business metrics
    retention_cost: float = 100.0  # Cost to retain a customer
    acquisition_cost: float = 500.0  # Cost to acquire a new customer
    false_positive_cost: float = 50.0  # Cost of false positive (unnecessary retention effort)
    
    def __post_init__(self):
        if self.secondary_metrics is None:
            self.secondary_metrics = ["precision", "recall", "f1", "precision_at_k"]


@dataclass
class ProjectConfig:
    """Main project configuration."""
    
    # Project info
    project_name: str = "Customer Churn Prediction"
    version: str = "1.0.0"
    description: str = "Research demo for customer churn prediction in banking"
    
    # Paths
    project_root: str = "."
    assets_path: str = "assets"
    logs_path: str = "logs"
    
    # Data config
    data: DataConfig = None
    
    # Model config
    model: ModelConfig = None
    
    # Evaluation config
    evaluation: EvaluationConfig = None
    
    def __post_init__(self):
        if self.data is None:
            self.data = DataConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.evaluation is None:
            self.evaluation = EvaluationConfig()


def load_config(config_path: str) -> ProjectConfig:
    """Load configuration from YAML file."""
    
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    # Convert nested dictionaries to config objects
    data_config = DataConfig(**config_dict.get('data', {}))
    model_config = ModelConfig(**config_dict.get('model', {}))
    eval_config = EvaluationConfig(**config_dict.get('evaluation', {}))
    
    # Create main config
    main_config_dict = {k: v for k, v in config_dict.items() 
                       if k not in ['data', 'model', 'evaluation']}
    
    return ProjectConfig(
        data=data_config,
        model=model_config,
        evaluation=eval_config,
        **main_config_dict
    )


def save_config(config: ProjectConfig, config_path: str) -> None:
    """Save configuration to YAML file."""
    
    config_dict = {
        'project_name': config.project_name,
        'version': config.version,
        'description': config.description,
        'project_root': config.project_root,
        'assets_path': config.assets_path,
        'logs_path': config.logs_path,
        'data': {
            'raw_data_path': config.data.raw_data_path,
            'processed_data_path': config.data.processed_data_path,
            'features_path': config.data.features_path,
            'test_size': config.data.test_size,
            'validation_size': config.data.validation_size,
            'random_state': config.data.random_state,
            'time_column': config.data.time_column,
            'train_start_date': config.data.train_start_date,
            'train_end_date': config.data.train_end_date,
            'test_start_date': config.data.test_start_date,
            'test_end_date': config.data.test_end_date,
            'lookback_window': config.data.lookback_window,
            'feature_window': config.data.feature_window,
            'label_window': config.data.label_window,
        },
        'model': {
            'model_type': config.model.model_type,
            'n_estimators': config.model.n_estimators,
            'max_depth': config.model.max_depth,
            'learning_rate': config.model.learning_rate,
            'subsample': config.model.subsample,
            'colsample_bytree': config.model.colsample_bytree,
            'early_stopping_rounds': config.model.early_stopping_rounds,
            'eval_metric': config.model.eval_metric,
            'random_state': config.model.random_state,
            'cv_folds': config.model.cv_folds,
            'cv_strategy': config.model.cv_strategy,
        },
        'evaluation': {
            'primary_metric': config.evaluation.primary_metric,
            'secondary_metrics': config.evaluation.secondary_metrics,
            'optimize_threshold': config.evaluation.optimize_threshold,
            'threshold_method': config.evaluation.threshold_method,
            'retention_cost': config.evaluation.retention_cost,
            'acquisition_cost': config.evaluation.acquisition_cost,
            'false_positive_cost': config.evaluation.false_positive_cost,
        }
    }
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
