"""
Main training script for customer churn prediction.
"""

import argparse
import logging
from pathlib import Path
import sys
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data.generator import CustomerDataGenerator
from features.engineering import FeatureEngineer
from models.churn_predictor import ChurnPredictor
from models.evaluation import ChurnEvaluator
from utils.config import ProjectConfig, load_config, save_config
from utils.reproducibility import ensure_reproducibility


def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("logs/training.log")
        ]
    )


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train customer churn prediction models")
    parser.add_argument("--config", type=str, default="configs/default.yaml", 
                       help="Path to configuration file")
    parser.add_argument("--data-dir", type=str, default="data/raw",
                       help="Directory containing raw data")
    parser.add_argument("--output-dir", type=str, default="models",
                       help="Directory to save trained models")
    parser.add_argument("--assets-dir", type=str, default="assets",
                       help="Directory to save evaluation assets")
    parser.add_argument("--log-level", type=str, default="INFO",
                       help="Logging level")
    parser.add_argument("--generate-data", action="store_true",
                       help="Generate synthetic data if not present")
    parser.add_argument("--n-customers", type=int, default=10000,
                       help="Number of customers to generate")
    
    args = parser.parse_args()
    
    # Setup
    ensure_reproducibility()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    # Create directories
    Path("logs").mkdir(exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.assets_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting customer churn prediction training")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data directory: {args.data_dir}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Load configuration
        if Path(args.config).exists():
            config = load_config(args.config)
            logger.info(f"Loaded configuration from {args.config}")
        else:
            logger.warning(f"Config file {args.config} not found, using defaults")
            config = ProjectConfig()
        
        # Generate or load data
        data_generator = CustomerDataGenerator(seed=config.data.random_state)
        
        if args.generate_data or not Path(args.data_dir).exists():
            logger.info("Generating synthetic customer data")
            datasets = data_generator.generate_complete_dataset(
                n_customers=args.n_customers,
                churn_rate=0.15
            )
            data_generator.save_datasets(datasets, args.data_dir)
        else:
            logger.info(f"Loading data from {args.data_dir}")
            datasets = data_generator.load_datasets(args.data_dir)
        
        # Feature engineering
        logger.info("Creating features")
        feature_engineer = FeatureEngineer(seed=config.data.random_state)
        features_df = feature_engineer.create_all_features(
            datasets, 
            lookback_days=config.data.lookback_window
        )
        
        # Save features
        features_path = Path(args.data_dir).parent / "processed" / "features.parquet"
        features_path.parent.mkdir(parents=True, exist_ok=True)
        feature_engineer.save_features(features_df, str(features_path))
        
        # Model training
        logger.info("Training models")
        predictor = ChurnPredictor(seed=config.model.random_state)
        
        # Prepare data
        X_train, X_val, X_test, y_train, y_val, y_test, feature_names = predictor.prepare_data(
            features_df,
            test_size=config.data.test_size,
            validation_size=config.data.validation_size
        )
        
        # Train models
        training_results = predictor.train_models(
            X_train, y_train, X_val, y_val,
            cv_folds=config.model.cv_folds
        )
        
        # Create ensemble
        ensemble_results = predictor.create_ensemble(
            X_train, y_train, X_val, y_val
        )
        
        # Model evaluation
        logger.info("Evaluating models")
        evaluator = ChurnEvaluator(seed=config.data.random_state)
        evaluator.set_business_parameters(
            retention_cost=config.evaluation.retention_cost,
            acquisition_cost=config.evaluation.acquisition_cost,
            false_positive_cost=config.evaluation.false_positive_cost
        )
        
        # Evaluate individual models
        for model_name in predictor.models.keys():
            y_pred, y_pred_proba = predictor.predict(X_test, model_name=model_name)
            evaluator.evaluate_model(y_test, y_pred, y_pred_proba, model_name)
        
        # Evaluate ensemble
        y_pred_ensemble, y_pred_proba_ensemble = predictor.predict(X_test, use_ensemble=True)
        evaluator.evaluate_model(y_test, y_pred_ensemble, y_pred_proba_ensemble, "ensemble")
        
        # Generate plots
        logger.info("Generating evaluation plots")
        
        # ROC curves
        for model_name in predictor.models.keys():
            y_pred, y_pred_proba = predictor.predict(X_test, model_name=model_name)
            roc_plot = evaluator.create_roc_plot(y_test, y_pred_proba, model_name)
            roc_plot.write_html(f"{args.assets_dir}/roc_curve_{model_name}.html")
        
        # Ensemble ROC curve
        roc_plot_ensemble = evaluator.create_roc_plot(y_test, y_pred_proba_ensemble, "ensemble")
        roc_plot_ensemble.write_html(f"{args.assets_dir}/roc_curve_ensemble.html")
        
        # Precision-Recall curves
        for model_name in predictor.models.keys():
            y_pred, y_pred_proba = predictor.predict(X_test, model_name=model_name)
            pr_plot = evaluator.create_precision_recall_plot(y_test, y_pred_proba, model_name)
            pr_plot.write_html(f"{args.assets_dir}/precision_recall_{model_name}.html")
        
        # Confusion matrices
        for model_name in predictor.models.keys():
            y_pred, y_pred_proba = predictor.predict(X_test, model_name=model_name)
            cm_plot = evaluator.create_confusion_matrix_plot(y_test, y_pred, model_name)
            cm_plot.write_html(f"{args.assets_dir}/confusion_matrix_{model_name}.html")
        
        # Feature importance
        for model_name in predictor.models.keys():
            feature_importance = predictor.get_feature_importance(model_name)
            if not feature_importance.empty:
                importance_plot = evaluator.create_feature_importance_plot(
                    feature_importance, model_name
                )
                importance_plot.write_html(f"{args.assets_dir}/feature_importance_{model_name}.html")
        
        # Save models
        logger.info(f"Saving models to {args.output_dir}")
        predictor.save_models(args.output_dir)
        
        # Generate evaluation report
        evaluator.generate_evaluation_report(args.assets_dir)
        
        # Model ranking
        ranking_df = evaluator.get_model_ranking()
        ranking_df.to_csv(f"{args.assets_dir}/model_ranking.csv", index=False)
        
        logger.info("Training completed successfully")
        logger.info(f"Best model: {predictor.best_model}")
        logger.info(f"Best AUC: {predictor.best_score:.4f}")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        print(f"Total customers: {len(features_df)}")
        print(f"Features created: {len(feature_engineer.feature_names)}")
        print(f"Best model: {predictor.best_model}")
        print(f"Best AUC: {predictor.best_score:.4f}")
        print(f"Models saved to: {args.output_dir}")
        print(f"Evaluation assets saved to: {args.assets_dir}")
        print("="*50)
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise


if __name__ == "__main__":
    main()
