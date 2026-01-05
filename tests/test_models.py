"""
Unit tests for customer churn prediction models.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import CustomerDataGenerator
from features.engineering import FeatureEngineer
from models.churn_predictor import ChurnPredictor
from models.evaluation import ChurnEvaluator
from utils.config import ProjectConfig, DataConfig, ModelConfig, EvaluationConfig
from utils.reproducibility import set_seed


class TestCustomerDataGenerator:
    """Test customer data generation."""
    
    def test_init(self):
        """Test generator initialization."""
        generator = CustomerDataGenerator(seed=42)
        assert generator.seed == 42
    
    def test_generate_customer_base(self):
        """Test customer base generation."""
        generator = CustomerDataGenerator(seed=42)
        df = generator.generate_customer_base(n_customers=100)
        
        assert len(df) == 100
        assert 'customer_id' in df.columns
        assert 'age' in df.columns
        assert 'gender' in df.columns
        assert 'income' in df.columns
        assert 'education' in df.columns
        assert 'region' in df.columns
        assert 'tenure_years' in df.columns
        assert 'account_type' in df.columns
        
        # Check data types and ranges
        assert df['age'].min() >= 18
        assert df['age'].max() <= 80
        assert df['income'].min() >= 20000
        assert df['income'].max() <= 500000
        assert df['tenure_years'].min() >= 0.1
        assert df['tenure_years'].max() <= 30
    
    def test_generate_transaction_history(self):
        """Test transaction history generation."""
        generator = CustomerDataGenerator(seed=42)
        customer_df = generator.generate_customer_base(n_customers=10)
        transaction_df = generator.generate_transaction_history(
            customer_df, start_date="2023-01-01", end_date="2023-12-31"
        )
        
        assert len(transaction_df) > 0
        assert 'customer_id' in transaction_df.columns
        assert 'transaction_date' in transaction_df.columns
        assert 'amount' in transaction_df.columns
        assert 'transaction_type' in transaction_df.columns
        assert 'channel' in transaction_df.columns
        
        # Check transaction types
        valid_types = ['debit', 'credit', 'transfer', 'payment']
        assert all(t in valid_types for t in transaction_df['transaction_type'].unique())
        
        # Check channels
        valid_channels = ['online', 'mobile', 'atm', 'branch']
        assert all(c in valid_channels for c in transaction_df['channel'].unique())
    
    def test_generate_complete_dataset(self):
        """Test complete dataset generation."""
        generator = CustomerDataGenerator(seed=42)
        datasets = generator.generate_complete_dataset(n_customers=50)
        
        assert 'customers' in datasets
        assert 'transactions' in datasets
        assert 'balances' in datasets
        assert 'labels' in datasets
        
        assert len(datasets['customers']) == 50
        assert len(datasets['transactions']) > 0
        assert len(datasets['balances']) > 0
        assert len(datasets['labels']) == 50


class TestFeatureEngineer:
    """Test feature engineering."""
    
    def test_init(self):
        """Test feature engineer initialization."""
        engineer = FeatureEngineer(seed=42)
        assert engineer.seed == 42
    
    def test_create_demographic_features(self):
        """Test demographic feature creation."""
        engineer = FeatureEngineer(seed=42)
        
        # Create sample customer data
        customer_data = {
            'customer_id': ['CUST_001', 'CUST_002'],
            'age': [25, 45],
            'gender': ['M', 'F'],
            'income': [50000, 75000],
            'education': ['Bachelor', 'Master'],
            'region': ['Urban', 'Suburban'],
            'tenure_years': [2.5, 8.0],
            'account_type': ['Basic', 'Premium']
        }
        customer_df = pd.DataFrame(customer_data)
        
        features = engineer.create_demographic_features(customer_df)
        
        assert 'age_group' in features.columns
        assert 'income_category' in features.columns
        assert 'is_high_income' in features.columns
        assert 'is_long_tenure' in features.columns
        assert 'is_female' in features.columns
        assert 'education_level' in features.columns
        assert 'region_code' in features.columns
        assert 'account_type_code' in features.columns


class TestChurnPredictor:
    """Test churn predictor."""
    
    def test_init(self):
        """Test predictor initialization."""
        predictor = ChurnPredictor(seed=42)
        assert predictor.seed == 42
        assert len(predictor.models) == 5  # 5 models initialized
        assert 'xgboost' in predictor.models
        assert 'lightgbm' in predictor.models
        assert 'random_forest' in predictor.models
        assert 'gradient_boosting' in predictor.models
        assert 'logistic_regression' in predictor.models
    
    def test_prepare_data(self):
        """Test data preparation."""
        predictor = ChurnPredictor(seed=42)
        
        # Create sample data
        n_samples = 100
        features_df = pd.DataFrame({
            'customer_id': [f'CUST_{i:03d}' for i in range(n_samples)],
            'age': np.random.randint(25, 65, n_samples),
            'income': np.random.uniform(30000, 100000, n_samples),
            'tenure_years': np.random.uniform(1, 10, n_samples),
            'churn': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
        })
        
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = predictor.prepare_data(
            features_df, test_size=0.2, validation_size=0.2
        )
        
        assert len(X_train) + len(X_val) + len(X_test) == n_samples
        assert len(y_train) + len(y_val) + len(y_test) == n_samples
        assert len(feature_names) == 3  # age, income, tenure_years
        
        # Check that splits are roughly correct
        assert abs(len(X_test) / n_samples - 0.2) < 0.05
        assert abs(len(X_val) / n_samples - 0.16) < 0.05  # 0.2 * 0.8
    
    def test_get_feature_importance(self):
        """Test feature importance extraction."""
        predictor = ChurnPredictor(seed=42)
        
        # Mock feature names
        predictor.feature_names = ['feature1', 'feature2', 'feature3']
        
        # Mock model with feature importance
        class MockModel:
            def __init__(self):
                self.feature_importances_ = np.array([0.5, 0.3, 0.2])
        
        predictor.models['xgboost'] = MockModel()
        predictor.best_model = 'xgboost'
        
        importance_df = predictor.get_feature_importance()
        
        assert len(importance_df) == 3
        assert 'feature' in importance_df.columns
        assert 'importance' in importance_df.columns
        assert importance_df['importance'].sum() == pytest.approx(1.0)


class TestChurnEvaluator:
    """Test churn evaluator."""
    
    def test_init(self):
        """Test evaluator initialization."""
        evaluator = ChurnEvaluator(seed=42)
        assert evaluator.seed == 42
        assert evaluator.retention_cost == 100.0
        assert evaluator.acquisition_cost == 500.0
        assert evaluator.false_positive_cost == 50.0
    
    def test_set_business_parameters(self):
        """Test business parameter setting."""
        evaluator = ChurnEvaluator(seed=42)
        evaluator.set_business_parameters(
            retention_cost=150.0,
            acquisition_cost=600.0,
            false_positive_cost=75.0
        )
        
        assert evaluator.retention_cost == 150.0
        assert evaluator.acquisition_cost == 600.0
        assert evaluator.false_positive_cost == 75.0
    
    def test_calculate_ml_metrics(self):
        """Test ML metrics calculation."""
        evaluator = ChurnEvaluator(seed=42)
        
        # Create sample data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.2, 0.9])
        
        metrics = evaluator.calculate_ml_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 'auc' in metrics
        assert 'average_precision' in metrics
        
        # Check metric ranges
        assert 0 <= metrics['accuracy'] <= 1
        assert 0 <= metrics['precision'] <= 1
        assert 0 <= metrics['recall'] <= 1
        assert 0 <= metrics['f1'] <= 1
        assert 0 <= metrics['auc'] <= 1
    
    def test_calculate_business_metrics(self):
        """Test business metrics calculation."""
        evaluator = ChurnEvaluator(seed=42)
        
        # Create sample data
        y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 0, 1, 0, 1])
        y_pred_proba = np.array([0.1, 0.9, 0.2, 0.4, 0.1, 0.8, 0.2, 0.9])
        
        metrics = evaluator.calculate_business_metrics(y_true, y_pred, y_pred_proba)
        
        assert 'total_customers' in metrics
        assert 'actual_churners' in metrics
        assert 'predicted_churners' in metrics
        assert 'retention_costs' in metrics
        assert 'total_costs' in metrics
        assert 'net_revenue' in metrics
        assert 'roi' in metrics
        
        assert metrics['total_customers'] == 8
        assert metrics['actual_churners'] == 4


class TestConfig:
    """Test configuration management."""
    
    def test_data_config(self):
        """Test data configuration."""
        config = DataConfig()
        
        assert config.test_size == 0.2
        assert config.validation_size == 0.2
        assert config.random_state == 42
        assert config.lookback_window == 30
        assert config.feature_window == 90
        assert config.label_window == 30
    
    def test_model_config(self):
        """Test model configuration."""
        config = ModelConfig()
        
        assert config.model_type == "xgboost"
        assert config.n_estimators == 100
        assert config.max_depth == 6
        assert config.learning_rate == 0.1
        assert config.cv_folds == 5
        assert config.cv_strategy == "time_series"
    
    def test_evaluation_config(self):
        """Test evaluation configuration."""
        config = EvaluationConfig()
        
        assert config.primary_metric == "auc"
        assert config.optimize_threshold == True
        assert config.threshold_method == "youden"
        assert config.retention_cost == 100.0
        assert config.acquisition_cost == 500.0
        assert config.false_positive_cost == 50.0
    
    def test_project_config(self):
        """Test project configuration."""
        config = ProjectConfig()
        
        assert config.project_name == "Customer Churn Prediction"
        assert config.version == "1.0.0"
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.evaluation, EvaluationConfig)


class TestReproducibility:
    """Test reproducibility utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.random()
        
        set_seed(42)
        val2 = np.random.random()
        
        assert val1 == val2
    
    def test_get_device(self):
        """Test device detection."""
        from utils.reproducibility import get_device
        
        device = get_device()
        assert device is not None
        assert str(device) in ['cpu', 'cuda', 'mps']


if __name__ == "__main__":
    pytest.main([__file__])
