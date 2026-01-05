"""
Streamlit demo for customer churn prediction.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from pathlib import Path
import sys
import joblib
import logging

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from models.churn_predictor import ChurnPredictor
from models.evaluation import ChurnEvaluator
from data.generator import CustomerDataGenerator
from features.engineering import FeatureEngineer

# Page configuration
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 0.5rem;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Disclaimer
st.markdown("""
<div class="warning-box">
    <h4>⚠️ Research Demo Disclaimer</h4>
    <p><strong>This is a research demonstration only and is not intended for investment advice or real-world banking decisions.</strong></p>
    <p>The models and predictions shown here are for educational purposes and may not be accurate or suitable for actual business use. 
    All results are hypothetical and should not be used to make financial decisions.</p>
</div>
""", unsafe_allow_html=True)

# Main header
st.markdown('<h1 class="main-header">🏦 Customer Churn Prediction Dashboard</h1>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.selectbox(
    "Select Page",
    ["Model Overview", "Customer Analysis", "Feature Importance", "Model Comparison", "About"]
)

# Initialize session state
if 'predictor' not in st.session_state:
    st.session_state.predictor = None
if 'evaluator' not in st.session_state:
    st.session_state.evaluator = None
if 'features_df' not in st.session_state:
    st.session_state.features_df = None

@st.cache_data
def load_models():
    """Load trained models."""
    try:
        predictor = ChurnPredictor()
        predictor.load_models("models")
        
        evaluator = ChurnEvaluator()
        
        return predictor, evaluator
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None

@st.cache_data
def load_features():
    """Load features data."""
    try:
        features_path = Path("data/processed/features.parquet")
        if features_path.exists():
            return pd.read_parquet(features_path)
        else:
            return None
    except Exception as e:
        st.error(f"Error loading features: {str(e)}")
        return None

def generate_sample_customer():
    """Generate a sample customer for demonstration."""
    np.random.seed(42)
    
    customer = {
        'age': np.random.randint(25, 65),
        'gender': np.random.choice(['M', 'F']),
        'income': np.random.lognormal(10.5, 0.8),
        'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD']),
        'region': np.random.choice(['Urban', 'Suburban', 'Rural']),
        'tenure_years': np.random.exponential(5),
        'account_type': np.random.choice(['Basic', 'Premium', 'Business', 'Student']),
        'total_transactions': np.random.poisson(15),
        'total_amount': np.random.lognormal(8, 1),
        'avg_transaction_amount': np.random.lognormal(6, 0.5),
        'transaction_frequency': np.random.uniform(0.1, 1.0),
        'debit_ratio': np.random.uniform(0.3, 0.8),
        'credit_ratio': np.random.uniform(0.1, 0.4),
        'online_ratio': np.random.uniform(0.4, 0.9),
        'mobile_ratio': np.random.uniform(0.2, 0.6),
        'avg_balance': np.random.lognormal(7, 1),
        'balance_volatility': np.random.lognormal(5, 0.5),
        'balance_trend': np.random.normal(0, 100),
        'risk_score': np.random.randint(0, 4),
    }
    
    return customer

if page == "Model Overview":
    st.header("📊 Model Performance Overview")
    
    # Load models
    predictor, evaluator = load_models()
    
    if predictor is None:
        st.error("Models not found. Please run the training script first.")
        st.code("python scripts/train.py --generate-data")
    else:
        # Model performance metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Best Model", predictor.best_model.title())
        
        with col2:
            st.metric("Best AUC", f"{predictor.best_score:.3f}")
        
        with col3:
            st.metric("Available Models", len(predictor.models))
        
        with col4:
            st.metric("Features", len(predictor.feature_names))
        
        # Model comparison chart
        st.subheader("Model Comparison")
        
        # Load evaluation results if available
        try:
            results_path = Path("assets/detailed_results.json")
            if results_path.exists():
                import json
                with open(results_path, 'r') as f:
                    results = json.load(f)
                
                # Create comparison chart
                model_names = list(results.keys())
                auc_scores = [results[model]['ml_metrics']['auc'] for model in model_names]
                f1_scores = [results[model]['ml_metrics']['f1'] for model in model_names]
                
                fig = make_subplots(
                    rows=1, cols=2,
                    subplot_titles=('AUC Scores', 'F1 Scores'),
                    specs=[[{"secondary_y": False}, {"secondary_y": False}]]
                )
                
                fig.add_trace(
                    go.Bar(x=model_names, y=auc_scores, name='AUC', marker_color='lightblue'),
                    row=1, col=1
                )
                
                fig.add_trace(
                    go.Bar(x=model_names, y=f1_scores, name='F1', marker_color='lightgreen'),
                    row=1, col=2
                )
                
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.info("Evaluation results not found. Run training to generate performance metrics.")
                
        except Exception as e:
            st.warning(f"Could not load evaluation results: {str(e)}")

elif page == "Customer Analysis":
    st.header("👤 Individual Customer Analysis")
    
    predictor, evaluator = load_models()
    
    if predictor is None:
        st.error("Models not found. Please run the training script first.")
    else:
        # Customer input form
        st.subheader("Customer Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.slider("Age", 18, 80, 35)
            gender = st.selectbox("Gender", ["M", "F"])
            income = st.number_input("Annual Income ($)", min_value=20000, max_value=500000, value=50000)
            education = st.selectbox("Education", ["High School", "Bachelor", "Master", "PhD"])
        
        with col2:
            region = st.selectbox("Region", ["Urban", "Suburban", "Rural"])
            tenure_years = st.slider("Years with Bank", 0.1, 30.0, 5.0)
            account_type = st.selectbox("Account Type", ["Basic", "Premium", "Business", "Student"])
            total_transactions = st.number_input("Total Transactions (90 days)", min_value=0, max_value=100, value=15)
        
        # Transaction behavior
        st.subheader("Transaction Behavior")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            avg_transaction_amount = st.number_input("Avg Transaction Amount ($)", min_value=1, max_value=10000, value=100)
            debit_ratio = st.slider("Debit Ratio", 0.0, 1.0, 0.6)
        
        with col2:
            online_ratio = st.slider("Online Usage Ratio", 0.0, 1.0, 0.7)
            mobile_ratio = st.slider("Mobile Usage Ratio", 0.0, 1.0, 0.4)
        
        with col3:
            avg_balance = st.number_input("Average Balance ($)", min_value=0, max_value=100000, value=5000)
            balance_volatility = st.number_input("Balance Volatility", min_value=0, max_value=10000, value=500)
        
        # Generate prediction
        if st.button("Predict Churn Risk"):
            # Create customer data
            customer_data = {
                'age': age,
                'gender': gender,
                'income': income,
                'education': education,
                'region': region,
                'tenure_years': tenure_years,
                'account_type': account_type,
                'total_transactions': total_transactions,
                'avg_transaction_amount': avg_transaction_amount,
                'debit_ratio': debit_ratio,
                'online_ratio': online_ratio,
                'mobile_ratio': mobile_ratio,
                'avg_balance': avg_balance,
                'balance_volatility': balance_volatility,
            }
            
            # Convert to DataFrame and encode categorical variables
            customer_df = pd.DataFrame([customer_data])
            
            # Simple encoding (in a real app, you'd use the same encoders from training)
            customer_df['is_female'] = (customer_df['gender'] == 'F').astype(int)
            customer_df['is_high_income'] = (customer_df['income'] > 75000).astype(int)
            customer_df['is_long_tenure'] = (customer_df['tenure_years'] > 5).astype(int)
            
            # Education encoding
            education_mapping = {'High School': 1, 'Bachelor': 2, 'Master': 3, 'PhD': 4}
            customer_df['education_level'] = customer_df['education'].map(education_mapping)
            
            # Region encoding
            region_mapping = {'Urban': 1, 'Suburban': 2, 'Rural': 3}
            customer_df['region_code'] = customer_df['region'].map(region_mapping)
            
            # Account type encoding
            account_type_mapping = {'Basic': 1, 'Student': 2, 'Premium': 3, 'Business': 4}
            customer_df['account_type_code'] = customer_df['account_type'].map(account_type_mapping)
            
            # Select features (simplified - in real app, use exact feature names from training)
            feature_cols = [
                'age', 'income', 'tenure_years', 'total_transactions',
                'avg_transaction_amount', 'debit_ratio', 'online_ratio',
                'mobile_ratio', 'avg_balance', 'balance_volatility',
                'is_female', 'is_high_income', 'is_long_tenure',
                'education_level', 'region_code', 'account_type_code'
            ]
            
            # Ensure all features exist
            for col in feature_cols:
                if col not in customer_df.columns:
                    customer_df[col] = 0
            
            X_customer = customer_df[feature_cols].values
            
            # Make prediction
            try:
                y_pred, y_pred_proba = predictor.predict(X_customer, use_ensemble=True)
                
                # Display results
                st.subheader("Prediction Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    churn_prob = y_pred_proba[0]
                    st.metric("Churn Probability", f"{churn_prob:.1%}")
                
                with col2:
                    risk_level = "High" if churn_prob > 0.7 else "Medium" if churn_prob > 0.3 else "Low"
                    st.metric("Risk Level", risk_level)
                
                with col3:
                    recommendation = "Retention Action Needed" if churn_prob > 0.5 else "Monitor"
                    st.metric("Recommendation", recommendation)
                
                # Risk factors
                st.subheader("Risk Factors")
                
                risk_factors = []
                if churn_prob > 0.5:
                    risk_factors.append("High churn probability")
                if tenure_years < 2:
                    risk_factors.append("New customer (high churn risk)")
                if income < 30000:
                    risk_factors.append("Low income")
                if debit_ratio > 0.8:
                    risk_factors.append("High debit ratio")
                if avg_balance < income * 0.05:
                    risk_factors.append("Low account balance")
                
                if risk_factors:
                    for factor in risk_factors:
                        st.warning(f"⚠️ {factor}")
                else:
                    st.success("✅ No major risk factors identified")
                
            except Exception as e:
                st.error(f"Prediction failed: {str(e)}")
        
        # Sample customer button
        if st.button("Load Sample Customer"):
            sample_customer = generate_sample_customer()
            st.info("Sample customer loaded. Adjust the sliders above to see different predictions.")

elif page == "Feature Importance":
    st.header("🔍 Feature Importance Analysis")
    
    predictor, evaluator = load_models()
    
    if predictor is None:
        st.error("Models not found. Please run the training script first.")
    else:
        # Model selection
        model_name = st.selectbox("Select Model", list(predictor.models.keys()))
        
        if st.button("Show Feature Importance"):
            try:
                feature_importance = predictor.get_feature_importance(model_name)
                
                if not feature_importance.empty:
                    # Top features
                    st.subheader(f"Top 20 Most Important Features - {model_name.title()}")
                    
                    top_features = feature_importance.head(20)
                    
                    fig = go.Figure(data=go.Bar(
                        x=top_features['importance'],
                        y=top_features['feature'],
                        orientation='h',
                        marker_color='lightblue'
                    ))
                    
                    fig.update_layout(
                        title=f'Feature Importance - {model_name.title()}',
                        xaxis_title='Importance',
                        yaxis_title='Features',
                        height=600
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Feature importance table
                    st.subheader("Feature Importance Table")
                    st.dataframe(feature_importance, use_container_width=True)
                    
                else:
                    st.warning(f"Feature importance not available for {model_name}")
                    
            except Exception as e:
                st.error(f"Error loading feature importance: {str(e)}")

elif page == "Model Comparison":
    st.header("⚖️ Model Comparison")
    
    # Load comparison data
    try:
        ranking_path = Path("assets/model_ranking.csv")
        if ranking_path.exists():
            ranking_df = pd.read_csv(ranking_path)
            
            st.subheader("Model Performance Ranking")
            st.dataframe(ranking_df, use_container_width=True)
            
            # Performance comparison chart
            st.subheader("Performance Comparison")
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('AUC Scores', 'F1 Scores', 'ROI', 'Net Revenue'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            fig.add_trace(
                go.Bar(x=ranking_df['Model'], y=ranking_df['AUC'], name='AUC', marker_color='lightblue'),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Bar(x=ranking_df['Model'], y=ranking_df['F1'], name='F1', marker_color='lightgreen'),
                row=1, col=2
            )
            
            fig.add_trace(
                go.Bar(x=ranking_df['Model'], y=ranking_df['ROI'], name='ROI', marker_color='orange'),
                row=2, col=1
            )
            
            fig.add_trace(
                go.Bar(x=ranking_df['Model'], y=ranking_df['Net Revenue'], name='Net Revenue', marker_color='purple'),
                row=2, col=2
            )
            
            fig.update_layout(height=800, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        else:
            st.info("Model comparison data not found. Run training to generate comparison metrics.")
            
    except Exception as e:
        st.error(f"Error loading comparison data: {str(e)}")

elif page == "About":
    st.header("ℹ️ About This Demo")
    
    st.markdown("""
    ## Customer Churn Prediction for Banks
    
    This is a research demonstration of machine learning models for predicting customer churn in banking.
    
    ### Features
    
    - **Multiple ML Models**: XGBoost, LightGBM, Random Forest, Gradient Boosting, Logistic Regression
    - **Ensemble Methods**: Weighted ensemble of top-performing models
    - **Comprehensive Evaluation**: Both ML metrics and business metrics
    - **Feature Engineering**: Demographic, transactional, and behavioral features
    - **Interactive Demo**: Streamlit-based web interface
    
    ### Model Performance
    
    The models are trained on synthetic customer data and evaluated using:
    - **ML Metrics**: AUC, Precision, Recall, F1-Score
    - **Business Metrics**: ROI, Cost Analysis, Revenue Impact
    
    ### Data Features
    
    - **Demographics**: Age, gender, income, education, region
    - **Account Info**: Tenure, account type, balance patterns
    - **Transaction Behavior**: Frequency, amounts, channels, types
    - **Risk Indicators**: Balance volatility, transaction patterns
    
    ### Disclaimer
    
    **This is a research demonstration only.** The models and predictions are for educational purposes and should not be used for actual banking decisions or investment advice.
    
    ### Technical Details
    
    - **Framework**: Python, scikit-learn, XGBoost, LightGBM
    - **Visualization**: Plotly, Streamlit
    - **Evaluation**: Comprehensive metrics and business analysis
    - **Reproducibility**: Deterministic seeding and version control
    
    ### Getting Started
    
    1. Run the training script to generate models:
       ```bash
       python scripts/train.py --generate-data
       ```
    
    2. Launch this demo:
       ```bash
       streamlit run demo/app.py
       ```
    
    ### Contact
    
    For questions about this research demo, please refer to the project documentation.
    """)
    
    # Technical specifications
    st.subheader("Technical Specifications")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Models Used:**
        - XGBoost
        - LightGBM
        - Random Forest
        - Gradient Boosting
        - Logistic Regression
        - Ensemble Methods
        """)
    
    with col2:
        st.markdown("""
        **Evaluation Metrics:**
        - AUC (Area Under Curve)
        - Precision & Recall
        - F1-Score
        - Business ROI
        - Cost Analysis
        """)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.8rem;">
    <p>Customer Churn Prediction Demo | Research Use Only | Not for Investment Advice</p>
</div>
""", unsafe_allow_html=True)
