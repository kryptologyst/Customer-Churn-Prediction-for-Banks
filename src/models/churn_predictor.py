"""
Advanced machine learning models for customer churn prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.calibration import CalibratedClassifierCV

import xgboost as xgb
import lightgbm as lgb

from utils.reproducibility import set_seed


class ChurnPredictor:
    """
    Advanced churn prediction model with multiple algorithms.
    
    This class provides a comprehensive churn prediction system with:
    - Multiple model algorithms (XGBoost, LightGBM, Random Forest, Logistic Regression)
    - Ensemble methods
    - Cross-validation
    - Feature importance analysis
    - Model calibration
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the churn predictor.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        set_seed(seed)
        self.logger = logging.getLogger(__name__)
        
        # Model storage
        self.models = {}
        self.scalers = {}
        self.feature_names = []
        self.best_model = None
        self.best_score = 0
        
        # Initialize models
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all available models."""
        
        # XGBoost
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            eval_metric='auc'
        )
        
        # LightGBM
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.seed,
            objective='binary',
            metric='auc',
            verbose=-1
        )
        
        # Random Forest
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.seed,
            n_jobs=-1
        )
        
        # Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            random_state=self.seed
        )
        
        # Logistic Regression
        self.models['logistic_regression'] = LogisticRegression(
            random_state=self.seed,
            max_iter=1000,
            C=1.0
        )
        
        self.logger.info(f"Initialized {len(self.models)} models")
    
    def prepare_data(
        self,
        features_df: pd.DataFrame,
        target_column: str = 'churn',
        test_size: float = 0.2,
        validation_size: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Prepare data for training and testing.
        
        Args:
            features_df: Features DataFrame
            target_column: Target column name
            test_size: Test set size
            validation_size: Validation set size
            
        Returns:
            Tuple of (X_train, X_val, X_test, y_train, y_val, y_test, feature_names)
        """
        self.logger.info("Preparing data for training")
        
        # Separate features and target
        feature_columns = [col for col in features_df.columns 
                          if col not in ['customer_id', target_column, 'churn_probability']]
        
        # Ensure all features are numeric
        numeric_features = []
        for col in feature_columns:
            if features_df[col].dtype in ['object', 'category']:
                # Skip non-numeric columns
                continue
            numeric_features.append(col)
        
        X = features_df[numeric_features].values
        y = features_df[target_column].values
        
        # Store feature names
        self.feature_names = numeric_features
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.seed, stratify=y
        )
        
        # Second split: train vs val
        val_size_adjusted = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_size_adjusted, 
            random_state=self.seed, stratify=y_temp
        )
        
        self.logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        self.logger.info(f"Churn rates - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}, Test: {y_test.mean():.3f}")
        
        return X_train, X_val, X_test, y_train, y_val, y_test, feature_columns
    
    def train_models(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        cv_folds: int = 5
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all models with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            cv_folds: Number of CV folds
            
        Returns:
            Dictionary with model performance metrics
        """
        self.logger.info(f"Training {len(self.models)} models with {cv_folds}-fold CV")
        
        results = {}
        
        for model_name, model in self.models.items():
            self.logger.info(f"Training {model_name}")
            
            # Scale features for models that need it
            if model_name in ['logistic_regression']:
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_val_scaled = scaler.transform(X_val)
                self.scalers[model_name] = scaler
            else:
                X_train_scaled = X_train
                X_val_scaled = X_val
            
            # Cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, 
                cv=cv_folds, scoring='roc_auc', n_jobs=-1
            )
            
            # Train on full training set
            model.fit(X_train_scaled, y_train)
            
            # Validation predictions
            y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            
            # Store results
            results[model_name] = {
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'val_auc': val_auc,
                'cv_scores': cv_scores.tolist()
            }
            
            self.logger.info(f"{model_name} - CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}, Val AUC: {val_auc:.4f}")
            
            # Update best model
            if val_auc > self.best_score:
                self.best_score = val_auc
                self.best_model = model_name
        
        self.logger.info(f"Best model: {self.best_model} (AUC: {self.best_score:.4f})")
        return results
    
    def create_ensemble(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        top_models: int = 3
    ) -> Dict[str, Any]:
        """
        Create ensemble model from top performing models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            top_models: Number of top models to include
            
        Returns:
            Ensemble performance metrics
        """
        self.logger.info(f"Creating ensemble from top {top_models} models")
        
        # Get top models by validation AUC
        model_scores = []
        for model_name, model in self.models.items():
            if model_name in self.scalers:
                X_val_scaled = self.scalers[model_name].transform(X_val)
            else:
                X_val_scaled = X_val
            
            y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            model_scores.append((model_name, val_auc))
        
        # Sort by AUC and select top models
        model_scores.sort(key=lambda x: x[1], reverse=True)
        top_model_names = [name for name, _ in model_scores[:top_models]]
        
        self.logger.info(f"Top {top_models} models: {top_model_names}")
        
        # Create ensemble predictions
        ensemble_predictions = np.zeros(len(X_val))
        weights = []
        
        for model_name in top_model_names:
            model = self.models[model_name]
            
            if model_name in self.scalers:
                X_val_scaled = self.scalers[model_name].transform(X_val)
            else:
                X_val_scaled = X_val
            
            y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            ensemble_predictions += y_val_pred_proba
            
            # Use validation AUC as weight
            val_auc = roc_auc_score(y_val, y_val_pred_proba)
            weights.append(val_auc)
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        # Weighted ensemble
        ensemble_predictions = np.zeros(len(X_val))
        for i, model_name in enumerate(top_model_names):
            model = self.models[model_name]
            
            if model_name in self.scalers:
                X_val_scaled = self.scalers[model_name].transform(X_val)
            else:
                X_val_scaled = X_val
            
            y_val_pred_proba = model.predict_proba(X_val_scaled)[:, 1]
            ensemble_predictions += weights[i] * y_val_pred_proba
        
        # Calculate ensemble performance
        ensemble_auc = roc_auc_score(y_val, ensemble_predictions)
        
        self.logger.info(f"Ensemble AUC: {ensemble_auc:.4f}")
        
        # Store ensemble info
        self.ensemble_models = top_model_names
        self.ensemble_weights = weights
        
        return {
            'ensemble_auc': ensemble_auc,
            'top_models': top_model_names,
            'weights': weights.tolist()
        }
    
    def predict(
        self,
        X: np.ndarray,
        model_name: Optional[str] = None,
        use_ensemble: bool = False
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using trained models.
        
        Args:
            X: Features to predict
            model_name: Specific model to use (if None, uses best model)
            use_ensemble: Whether to use ensemble prediction
            
        Returns:
            Tuple of (predictions, prediction_probabilities)
        """
        if use_ensemble and hasattr(self, 'ensemble_models'):
            # Ensemble prediction
            ensemble_predictions = np.zeros(len(X))
            
            for i, model_name in enumerate(self.ensemble_models):
                model = self.models[model_name]
                
                if model_name in self.scalers:
                    X_scaled = self.scalers[model_name].transform(X)
                else:
                    X_scaled = X
                
                y_pred_proba = model.predict_proba(X_scaled)[:, 1]
                ensemble_predictions += self.ensemble_weights[i] * y_pred_proba
            
            y_pred = (ensemble_predictions > 0.5).astype(int)
            return y_pred, ensemble_predictions
        
        else:
            # Single model prediction
            if model_name is None:
                model_name = self.best_model
            
            model = self.models[model_name]
            
            if model_name in self.scalers:
                X_scaled = self.scalers[model_name].transform(X)
            else:
                X_scaled = X
            
            y_pred = model.predict(X_scaled)
            y_pred_proba = model.predict_proba(X_scaled)[:, 1]
            
            return y_pred, y_pred_proba
    
    def get_feature_importance(self, model_name: Optional[str] = None) -> pd.DataFrame:
        """
        Get feature importance from trained model.
        
        Args:
            model_name: Model to get importance from (if None, uses best model)
            
        Returns:
            DataFrame with feature importance
        """
        if model_name is None:
            model_name = self.best_model
        
        model = self.models[model_name]
        
        # Get feature importance
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance = np.abs(model.coef_[0])
        else:
            self.logger.warning(f"Model {model_name} does not support feature importance")
            return pd.DataFrame()
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_models(self, output_dir: str) -> None:
        """
        Save trained models to disk.
        
        Args:
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save models
        for model_name, model in self.models.items():
            model_path = output_path / f"{model_name}_model.joblib"
            joblib.dump(model, model_path)
        
        # Save scalers
        for scaler_name, scaler in self.scalers.items():
            scaler_path = output_path / f"{scaler_name}_scaler.joblib"
            joblib.dump(scaler, scaler_path)
        
        # Save metadata
        metadata = {
            'best_model': self.best_model,
            'best_score': self.best_score,
            'feature_names': self.feature_names,
            'ensemble_models': getattr(self, 'ensemble_models', None),
            'ensemble_weights': getattr(self, 'ensemble_weights', None).tolist() if hasattr(self, 'ensemble_weights') else None
        }
        
        metadata_path = output_path / "model_metadata.joblib"
        joblib.dump(metadata, metadata_path)
        
        self.logger.info(f"Saved models to {output_path}")
    
    def load_models(self, input_dir: str) -> None:
        """
        Load trained models from disk.
        
        Args:
            input_dir: Input directory
        """
        input_path = Path(input_dir)
        
        # Load metadata
        metadata_path = input_path / "model_metadata.joblib"
        if metadata_path.exists():
            metadata = joblib.load(metadata_path)
            self.best_model = metadata['best_model']
            self.best_score = metadata['best_score']
            self.feature_names = metadata['feature_names']
            
            if metadata['ensemble_models']:
                self.ensemble_models = metadata['ensemble_models']
                self.ensemble_weights = np.array(metadata['ensemble_weights'])
        
        # Load models
        for model_name in self.models.keys():
            model_path = input_path / f"{model_name}_model.joblib"
            if model_path.exists():
                self.models[model_name] = joblib.load(model_path)
        
        # Load scalers
        for scaler_name in self.scalers.keys():
            scaler_path = input_path / f"{scaler_name}_scaler.joblib"
            if scaler_path.exists():
                self.scalers[scaler_name] = joblib.load(scaler_path)
        
        self.logger.info(f"Loaded models from {input_path}")
