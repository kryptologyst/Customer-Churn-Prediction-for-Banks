# Customer Churn Prediction for Banks

**⚠️ RESEARCH DEMO ONLY - NOT FOR INVESTMENT ADVICE**

This project is a comprehensive research demonstration of machine learning models for predicting customer churn in banking. It provides a complete pipeline from data generation to model deployment with interactive visualization.

## Important Disclaimer

**This is a research demonstration only and is not intended for investment advice or real-world banking decisions.** The models and predictions shown here are for educational purposes and may not be accurate or suitable for actual business use. All results are hypothetical and should not be used to make financial decisions.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Model Performance](#model-performance)
- [Evaluation Metrics](#evaluation-metrics)
- [Interactive Demo](#interactive-demo)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

## Features

### Machine Learning Models
- **XGBoost**: Gradient boosting with advanced regularization
- **LightGBM**: Fast gradient boosting framework
- **Random Forest**: Ensemble of decision trees
- **Gradient Boosting**: Traditional gradient boosting
- **Logistic Regression**: Linear baseline model
- **Ensemble Methods**: Weighted combination of top models

### Data Processing
- **Synthetic Data Generation**: Realistic customer data simulation
- **Feature Engineering**: Comprehensive feature creation
- **Time-based Splits**: Proper temporal data handling
- **Leakage Prevention**: Strict separation of feature and label windows

### Evaluation Framework
- **ML Metrics**: AUC, Precision, Recall, F1-Score, Precision@K
- **Business Metrics**: ROI, Cost Analysis, Revenue Impact
- **Threshold Optimization**: Multiple optimization strategies
- **Calibration Analysis**: Model reliability assessment

### Interactive Demo
- **Streamlit Dashboard**: Web-based interface
- **Customer Analysis**: Individual customer risk assessment
- **Feature Importance**: Model interpretability
- **Model Comparison**: Performance benchmarking

## Installation

### Prerequisites
- Python 3.10 or higher
- pip or conda package manager

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Customer-Churn-Prediction-for-Banks.git
cd Customer-Churn-Prediction-for-Banks

# Install dependencies
pip install -r requirements.txt

# Or install in development mode
pip install -e .
```

### Optional Dependencies

```bash
# For development
pip install -e ".[dev]"

# For experiment tracking
pip install -e ".[tracking]"

# For model serving
pip install -e ".[serving]"
```

## Quick Start

### 1. Generate Data and Train Models

```bash
# Generate synthetic data and train models
python scripts/train.py --generate-data --n-customers 10000

# Or use existing data
python scripts/train.py --data-dir data/raw
```

### 2. Launch Interactive Demo

```bash
# Start Streamlit dashboard
streamlit run demo/app.py
```

### 3. View Results

- Models saved to: `models/`
- Evaluation plots: `assets/`
- Training logs: `logs/`

## 📁 Project Structure

```
customer-churn-prediction/
├── src/                          # Source code
│   ├── data/                     # Data processing
│   │   └── generator.py          # Synthetic data generation
│   ├── features/                 # Feature engineering
│   │   └── engineering.py        # Feature creation
│   ├── models/                   # ML models
│   │   ├── churn_predictor.py    # Model training
│   │   └── evaluation.py        # Model evaluation
│   └── utils/                    # Utilities
│       ├── config.py             # Configuration management
│       └── reproducibility.py   # Reproducibility tools
├── scripts/                      # Training scripts
│   └── train.py                 # Main training script
├── configs/                      # Configuration files
│   └── default.yaml             # Default configuration
├── demo/                         # Interactive demo
│   └── app.py                   # Streamlit application
├── data/                         # Data storage
│   ├── raw/                     # Raw data
│   ├── processed/               # Processed data
│   └── features/                # Feature data
├── models/                       # Trained models
├── assets/                       # Evaluation assets
├── logs/                         # Training logs
├── tests/                        # Unit tests
├── notebooks/                    # Jupyter notebooks
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Project configuration
└── README.md                     # This file
```

## Usage

### Training Models

```bash
# Basic training with default settings
python scripts/train.py

# Generate new data and train
python scripts/train.py --generate-data --n-customers 15000

# Custom configuration
python scripts/train.py --config configs/custom.yaml

# Specify output directories
python scripts/train.py --output-dir models/v2 --assets-dir assets/v2
```

### Configuration

The project uses YAML configuration files. Key parameters:

```yaml
# Data configuration
data:
  test_size: 0.2
  validation_size: 0.2
  lookback_window: 90
  random_state: 42

# Model configuration
model:
  model_type: "xgboost"
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  cv_folds: 5

# Evaluation configuration
evaluation:
  retention_cost: 100.0
  acquisition_cost: 500.0
  false_positive_cost: 50.0
```

### Programmatic Usage

```python
from src.data.generator import CustomerDataGenerator
from src.features.engineering import FeatureEngineer
from src.models.churn_predictor import ChurnPredictor
from src.models.evaluation import ChurnEvaluator

# Generate data
generator = CustomerDataGenerator(seed=42)
datasets = generator.generate_complete_dataset(n_customers=10000)

# Create features
engineer = FeatureEngineer(seed=42)
features_df = engineer.create_all_features(datasets)

# Train models
predictor = ChurnPredictor(seed=42)
X_train, X_val, X_test, y_train, y_val, y_test, feature_names = predictor.prepare_data(features_df)
predictor.train_models(X_train, y_train, X_val, y_val)

# Evaluate
evaluator = ChurnEvaluator(seed=42)
y_pred, y_pred_proba = predictor.predict(X_test, use_ensemble=True)
results = evaluator.evaluate_model(y_test, y_pred, y_pred_proba, "ensemble")
```

## Model Performance

### Typical Performance Metrics

| Model | AUC | Precision | Recall | F1-Score | ROI |
|-------|-----|-----------|--------|----------|-----|
| XGBoost | 0.85+ | 0.75+ | 0.70+ | 0.72+ | 2.5+ |
| LightGBM | 0.84+ | 0.74+ | 0.69+ | 0.71+ | 2.4+ |
| Random Forest | 0.82+ | 0.72+ | 0.67+ | 0.69+ | 2.2+ |
| Ensemble | 0.86+ | 0.76+ | 0.72+ | 0.74+ | 2.6+ |

*Note: Performance may vary based on data characteristics and hyperparameters.*

### Feature Importance

Top features typically include:
- Account balance patterns
- Transaction frequency
- Customer tenure
- Income level
- Transaction channel preferences
- Risk indicators

## Evaluation Metrics

### Machine Learning Metrics
- **AUC**: Area Under the ROC Curve
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **Precision@K**: Precision for top K predictions

### Business Metrics
- **ROI**: Return on investment from retention efforts
- **Cost Analysis**: Total costs vs. benefits
- **Revenue Impact**: Net revenue from retention
- **Cost per Customer**: Average cost per customer

### Model Calibration
- **Calibration Curve**: Reliability of probability estimates
- **Brier Score**: Calibration quality metric
- **Threshold Optimization**: Optimal decision thresholds

## Interactive Demo

The Streamlit demo provides:

### Model Overview
- Performance metrics dashboard
- Model comparison charts
- Training statistics

### Customer Analysis
- Individual customer risk assessment
- Interactive parameter adjustment
- Risk factor identification
- Retention recommendations

### Feature Importance
- Model interpretability
- Top feature visualization
- Feature importance tables

### Model Comparison
- Performance benchmarking
- Multi-metric comparison
- Ranking analysis

## Configuration

### Environment Variables

```bash
# Optional: Set custom paths
export CHURN_DATA_DIR="data/custom"
export CHURN_MODEL_DIR="models/custom"
export CHURN_ASSETS_DIR="assets/custom"
```

### Configuration Files

- `configs/default.yaml`: Default configuration
- `configs/production.yaml`: Production settings
- `configs/experiment.yaml`: Experimental settings

### Logging

Logs are written to:
- Console output
- `logs/training.log`
- `logs/evaluation.log`

## Testing

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test file
pytest tests/test_models.py
```

## Development

### Code Quality

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Type checking
mypy src/
```

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Development Guidelines

- Follow PEP 8 style guidelines
- Add type hints to all functions
- Write comprehensive docstrings
- Include unit tests for new features
- Update documentation as needed

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## Version History

- **v1.0.0**: Initial release with comprehensive churn prediction pipeline
- **v1.1.0**: Added ensemble methods and improved evaluation
- **v1.2.0**: Enhanced Streamlit demo and feature engineering

---

**Remember: This is a research demonstration only. Do not use for actual banking decisions or investment advice.**
# Customer-Churn-Prediction-for-Banks
