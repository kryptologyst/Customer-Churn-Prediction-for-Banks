"""
Comprehensive evaluation metrics for customer churn prediction.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, roc_curve,
    confusion_matrix, classification_report, precision_score,
    recall_score, f1_score, accuracy_score, average_precision_score
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.reproducibility import set_seed


class ChurnEvaluator:
    """
    Comprehensive evaluation system for churn prediction models.
    
    This class provides both ML metrics and business metrics for
    evaluating churn prediction models, including:
    - Standard ML metrics (AUC, precision, recall, F1)
    - Business metrics (cost analysis, ROI)
    - Calibration analysis
    - Feature importance analysis
    - Visualization tools
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the evaluator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        set_seed(seed)
        self.logger = logging.getLogger(__name__)
        
        # Business parameters
        self.retention_cost = 100.0  # Cost to retain a customer
        self.acquisition_cost = 500.0  # Cost to acquire a new customer
        self.false_positive_cost = 50.0  # Cost of false positive (unnecessary retention effort)
        
        # Results storage
        self.results = {}
        self.plots = {}
    
    def set_business_parameters(
        self,
        retention_cost: float = 100.0,
        acquisition_cost: float = 500.0,
        false_positive_cost: float = 50.0
    ) -> None:
        """
        Set business parameters for cost analysis.
        
        Args:
            retention_cost: Cost to retain a customer
            acquisition_cost: Cost to acquire a new customer
            false_positive_cost: Cost of false positive (unnecessary retention effort)
        """
        self.retention_cost = retention_cost
        self.acquisition_cost = acquisition_cost
        self.false_positive_cost = false_positive_cost
        
        self.logger.info(f"Updated business parameters - Retention: ${retention_cost}, "
                        f"Acquisition: ${acquisition_cost}, FP Cost: ${false_positive_cost}")
    
    def calculate_ml_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict[str, float]:
        """
        Calculate standard ML metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            
        Returns:
            Dictionary of ML metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC metrics
        metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)
        
        # Precision at different thresholds
        for k in [100, 500, 1000]:
            if len(y_true) >= k:
                # Get top k predictions by probability
                top_k_indices = np.argsort(y_pred_proba)[-k:]
                top_k_true = y_true[top_k_indices]
                metrics[f'precision_at_{k}'] = top_k_true.mean()
        
        # Confusion matrix components
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        metrics['true_positives'] = int(tp)
        
        # Additional metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        return metrics
    
    def calculate_business_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        threshold: float = 0.5
    ) -> Dict[str, float]:
        """
        Calculate business-relevant metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            threshold: Classification threshold
            
        Returns:
            Dictionary of business metrics
        """
        # Recalculate predictions with threshold
        y_pred_thresh = (y_pred_proba >= threshold).astype(int)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_thresh).ravel()
        
        # Business calculations
        total_customers = len(y_true)
        actual_churners = y_true.sum()
        predicted_churners = y_pred_thresh.sum()
        
        # Cost analysis
        retention_costs = predicted_churners * self.retention_cost
        false_positive_costs = fp * self.false_positive_cost
        acquisition_costs = fn * self.acquisition_cost  # Lost customers we didn't retain
        
        total_costs = retention_costs + false_positive_costs + acquisition_costs
        
        # Revenue analysis (simplified)
        # Assume each retained customer generates $1000 in revenue
        customer_lifetime_value = 1000.0
        retained_revenue = tp * customer_lifetime_value
        lost_revenue = fn * customer_lifetime_value
        
        net_revenue = retained_revenue - lost_revenue
        roi = (net_revenue - total_costs) / total_costs if total_costs > 0 else 0
        
        # Business metrics
        metrics = {
            'total_customers': int(total_customers),
            'actual_churners': int(actual_churners),
            'predicted_churners': int(predicted_churners),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_negatives': int(tn),
            'retention_costs': retention_costs,
            'false_positive_costs': false_positive_costs,
            'acquisition_costs': acquisition_costs,
            'total_costs': total_costs,
            'retained_revenue': retained_revenue,
            'lost_revenue': lost_revenue,
            'net_revenue': net_revenue,
            'roi': roi,
            'cost_per_customer': total_costs / total_customers,
            'revenue_per_customer': net_revenue / total_customers
        }
        
        return metrics
    
    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        method: str = 'youden'
    ) -> Tuple[float, Dict[str, float]]:
        """
        Optimize classification threshold.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            method: Optimization method ('youden', 'f1', 'precision_recall')
            
        Returns:
            Tuple of (optimal_threshold, metrics_at_threshold)
        """
        if method == 'youden':
            # Youden's J statistic (maximize sensitivity + specificity - 1)
            fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
            youden_j = tpr - fpr
            optimal_idx = np.argmax(youden_j)
            optimal_threshold = thresholds[optimal_idx]
            
        elif method == 'f1':
            # Maximize F1 score
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx]
            
        elif method == 'precision_recall':
            # Maximize precision * recall
            precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
            precision_recall_score = precision * recall
            optimal_idx = np.argmax(precision_recall_score)
            optimal_threshold = thresholds[optimal_idx]
            
        else:
            raise ValueError(f"Unknown optimization method: {method}")
        
        # Calculate metrics at optimal threshold
        y_pred_optimal = (y_pred_proba >= optimal_threshold).astype(int)
        ml_metrics = self.calculate_ml_metrics(y_true, y_pred_optimal, y_pred_proba)
        business_metrics = self.calculate_business_metrics(y_true, y_pred_optimal, y_pred_proba, optimal_threshold)
        
        self.logger.info(f"Optimal threshold ({method}): {optimal_threshold:.4f}")
        
        return optimal_threshold, {**ml_metrics, **business_metrics}
    
    def evaluate_model(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model",
        optimize_threshold: bool = True
    ) -> Dict[str, Any]:
        """
        Comprehensive model evaluation.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            optimize_threshold: Whether to optimize threshold
            
        Returns:
            Dictionary with all evaluation results
        """
        self.logger.info(f"Evaluating {model_name}")
        
        # Standard ML metrics
        ml_metrics = self.calculate_ml_metrics(y_true, y_pred, y_pred_proba)
        
        # Business metrics
        business_metrics = self.calculate_business_metrics(y_true, y_pred, y_pred_proba)
        
        # Threshold optimization
        threshold_results = {}
        if optimize_threshold:
            optimal_threshold, optimal_metrics = self.optimize_threshold(y_true, y_pred_proba)
            threshold_results = {
                'optimal_threshold': optimal_threshold,
                'optimal_metrics': optimal_metrics
            }
        
        # Store results
        results = {
            'model_name': model_name,
            'ml_metrics': ml_metrics,
            'business_metrics': business_metrics,
            'threshold_optimization': threshold_results,
            'data_summary': {
                'total_samples': len(y_true),
                'positive_samples': int(y_true.sum()),
                'negative_samples': int(len(y_true) - y_true.sum()),
                'positive_rate': float(y_true.mean())
            }
        }
        
        self.results[model_name] = results
        
        self.logger.info(f"{model_name} - AUC: {ml_metrics['auc']:.4f}, "
                        f"F1: {ml_metrics['f1']:.4f}, ROI: {business_metrics['roi']:.2f}")
        
        return results
    
    def create_roc_plot(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model"
    ) -> go.Figure:
        """
        Create ROC curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
        auc_score = roc_auc_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr,
            y=tpr,
            mode='lines',
            name=f'{model_name} (AUC = {auc_score:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Random classifier
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=600,
            height=500
        )
        
        return fig
    
    def create_precision_recall_plot(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model"
    ) -> go.Figure:
        """
        Create precision-recall curve plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
        avg_precision = average_precision_score(y_true, y_pred_proba)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=recall,
            y=precision,
            mode='lines',
            name=f'{model_name} (AP = {avg_precision:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Random classifier baseline
        baseline = y_true.mean()
        fig.add_hline(
            y=baseline,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Random Classifier (AP = {baseline:.3f})"
        )
        
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=600,
            height=500
        )
        
        return fig
    
    def create_calibration_plot(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        model_name: str = "model",
        n_bins: int = 10
    ) -> go.Figure:
        """
        Create calibration plot.
        
        Args:
            y_true: True labels
            y_pred_proba: Predicted probabilities
            model_name: Name of the model
            n_bins: Number of bins for calibration
            
        Returns:
            Plotly figure
        """
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_pred_proba, n_bins=n_bins
        )
        
        fig = go.Figure()
        
        # Calibration curve
        fig.add_trace(go.Scatter(
            x=mean_predicted_value,
            y=fraction_of_positives,
            mode='lines+markers',
            name=f'{model_name}',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Perfect calibration line
        fig.add_trace(go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            name='Perfect Calibration',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title=f'Calibration Plot - {model_name}',
            xaxis_title='Mean Predicted Probability',
            yaxis_title='Fraction of Positives',
            width=600,
            height=500
        )
        
        return fig
    
    def create_confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        model_name: str = "model"
    ) -> go.Figure:
        """
        Create confusion matrix heatmap.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            model_name: Name of the model
            
        Returns:
            Plotly figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        fig = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Predicted No Churn', 'Predicted Churn'],
            y=['Actual No Churn', 'Actual Churn'],
            colorscale='Blues',
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 16},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title=f'Confusion Matrix - {model_name}',
            width=500,
            height=400
        )
        
        return fig
    
    def create_feature_importance_plot(
        self,
        feature_importance: pd.DataFrame,
        model_name: str = "model",
        top_n: int = 20
    ) -> go.Figure:
        """
        Create feature importance plot.
        
        Args:
            feature_importance: DataFrame with feature importance
            model_name: Name of the model
            top_n: Number of top features to show
            
        Returns:
            Plotly figure
        """
        # Get top features
        top_features = feature_importance.head(top_n)
        
        fig = go.Figure(data=go.Bar(
            x=top_features['importance'],
            y=top_features['feature'],
            orientation='h',
            marker=dict(color='lightblue')
        ))
        
        fig.update_layout(
            title=f'Top {top_n} Feature Importance - {model_name}',
            xaxis_title='Importance',
            yaxis_title='Features',
            width=800,
            height=600
        )
        
        return fig
    
    def generate_evaluation_report(
        self,
        output_dir: str = "assets"
    ) -> None:
        """
        Generate comprehensive evaluation report.
        
        Args:
            output_dir: Output directory for plots and reports
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Generating evaluation report in {output_path}")
        
        # Create summary DataFrame
        summary_data = []
        for model_name, results in self.results.items():
            ml_metrics = results['ml_metrics']
            business_metrics = results['business_metrics']
            
            summary_data.append({
                'Model': model_name,
                'AUC': ml_metrics['auc'],
                'Precision': ml_metrics['precision'],
                'Recall': ml_metrics['recall'],
                'F1': ml_metrics['f1'],
                'ROI': business_metrics['roi'],
                'Total Cost': business_metrics['total_costs'],
                'Net Revenue': business_metrics['net_revenue']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(output_path / "model_comparison.csv", index=False)
        
        # Save detailed results
        import json
        with open(output_path / "detailed_results.json", 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        self.logger.info("Evaluation report generated successfully")
    
    def get_model_ranking(self) -> pd.DataFrame:
        """
        Get model ranking by different metrics.
        
        Returns:
            DataFrame with model rankings
        """
        if not self.results:
            return pd.DataFrame()
        
        ranking_data = []
        for model_name, results in self.results.items():
            ml_metrics = results['ml_metrics']
            business_metrics = results['business_metrics']
            
            ranking_data.append({
                'Model': model_name,
                'AUC': ml_metrics['auc'],
                'F1': ml_metrics['f1'],
                'ROI': business_metrics['roi'],
                'Net Revenue': business_metrics['net_revenue']
            })
        
        ranking_df = pd.DataFrame(ranking_data)
        
        # Add rankings
        ranking_df['AUC_Rank'] = ranking_df['AUC'].rank(ascending=False)
        ranking_df['F1_Rank'] = ranking_df['F1'].rank(ascending=False)
        ranking_df['ROI_Rank'] = ranking_df['ROI'].rank(ascending=False)
        ranking_df['Revenue_Rank'] = ranking_df['Net Revenue'].rank(ascending=False)
        
        # Overall rank (average of all ranks)
        ranking_df['Overall_Rank'] = (
            ranking_df['AUC_Rank'] + ranking_df['F1_Rank'] + 
            ranking_df['ROI_Rank'] + ranking_df['Revenue_Rank']
        ) / 4
        
        return ranking_df.sort_values('Overall_Rank')
