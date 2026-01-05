"""
Feature engineering for customer churn prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

from utils.reproducibility import set_seed


class FeatureEngineer:
    """
    Feature engineering for customer churn prediction.
    
    This class creates comprehensive features from customer data including:
    - Demographic features
    - Transaction behavior features
    - Balance patterns
    - Temporal features
    - Risk indicators
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the feature engineer.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        set_seed(seed)
        self.logger = logging.getLogger(__name__)
        self.feature_names = []
    
    def create_demographic_features(self, customer_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create demographic features from customer data.
        
        Args:
            customer_df: Customer base data
            
        Returns:
            DataFrame with demographic features
        """
        self.logger.info("Creating demographic features")
        
        features = customer_df.copy()
        
        # Age groups
        features['age_group'] = pd.cut(
            features['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        
        # Income categories
        features['income_category'] = pd.cut(
            features['income'],
            bins=[0, 30000, 50000, 75000, 100000, float('inf')],
            labels=['Low', 'Lower-Mid', 'Mid', 'Upper-Mid', 'High']
        )
        
        # Income per year of tenure
        features['income_per_tenure'] = features['income'] / (features['tenure_years'] + 1)
        
        # Binary features
        features['is_high_income'] = (features['income'] > features['income'].quantile(0.75)).astype(int)
        features['is_long_tenure'] = (features['tenure_years'] > features['tenure_years'].quantile(0.75)).astype(int)
        features['is_young'] = (features['age'] < 30).astype(int)
        features['is_senior'] = (features['age'] > 60).astype(int)
        
        # Gender encoding
        features['is_female'] = (features['gender'] == 'F').astype(int)
        
        # Education encoding
        education_mapping = {
            'High School': 1,
            'Bachelor': 2,
            'Master': 3,
            'PhD': 4
        }
        features['education_level'] = features['education'].map(education_mapping)
        
        # Region encoding
        region_mapping = {
            'Urban': 1,
            'Suburban': 2,
            'Rural': 3
        }
        features['region_code'] = features['region'].map(region_mapping)
        
        # Account type encoding
        account_type_mapping = {
            'Basic': 1,
            'Student': 2,
            'Premium': 3,
            'Business': 4
        }
        features['account_type_code'] = features['account_type'].map(account_type_mapping)
        
        self.logger.info(f"Created {len(features.columns)} demographic features")
        return features
    
    def create_transaction_features(
        self,
        customer_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Create transaction-based features.
        
        Args:
            customer_df: Customer base data
            transaction_df: Transaction history
            lookback_days: Number of days to look back for features
            
        Returns:
            DataFrame with transaction features
        """
        self.logger.info(f"Creating transaction features (lookback: {lookback_days} days)")
        
        features = customer_df[['customer_id']].copy()
        
        # Calculate cutoff date
        max_date = transaction_df['transaction_date'].max()
        cutoff_date = max_date - timedelta(days=lookback_days)
        
        for customer_id in customer_df['customer_id']:
            customer_transactions = transaction_df[
                (transaction_df['customer_id'] == customer_id) &
                (transaction_df['transaction_date'] >= cutoff_date)
            ]
            
            if len(customer_transactions) == 0:
                # No transactions in lookback period
                features.loc[features['customer_id'] == customer_id, 'total_transactions'] = 0
                features.loc[features['customer_id'] == customer_id, 'total_amount'] = 0
                features.loc[features['customer_id'] == customer_id, 'avg_transaction_amount'] = 0
                features.loc[features['customer_id'] == customer_id, 'transaction_frequency'] = 0
                features.loc[features['customer_id'] == customer_id, 'debit_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'credit_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'transfer_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'payment_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'online_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'mobile_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'atm_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'branch_ratio'] = 0
                features.loc[features['customer_id'] == customer_id, 'days_since_last_transaction'] = lookback_days
                continue
            
            # Basic transaction metrics
            total_transactions = len(customer_transactions)
            total_amount = customer_transactions['amount'].sum()
            avg_transaction_amount = customer_transactions['amount'].mean()
            transaction_frequency = total_transactions / lookback_days
            
            # Transaction type ratios
            transaction_types = customer_transactions['transaction_type'].value_counts()
            total_txns = len(customer_transactions)
            
            debit_ratio = transaction_types.get('debit', 0) / total_txns
            credit_ratio = transaction_types.get('credit', 0) / total_txns
            transfer_ratio = transaction_types.get('transfer', 0) / total_txns
            payment_ratio = transaction_types.get('payment', 0) / total_txns
            
            # Channel ratios
            channels = customer_transactions['channel'].value_counts()
            online_ratio = channels.get('online', 0) / total_txns
            mobile_ratio = channels.get('mobile', 0) / total_txns
            atm_ratio = channels.get('atm', 0) / total_txns
            branch_ratio = channels.get('branch', 0) / total_txns
            
            # Days since last transaction
            days_since_last = (max_date - customer_transactions['transaction_date'].max()).days
            
            # Update features
            mask = features['customer_id'] == customer_id
            features.loc[mask, 'total_transactions'] = total_transactions
            features.loc[mask, 'total_amount'] = total_amount
            features.loc[mask, 'avg_transaction_amount'] = avg_transaction_amount
            features.loc[mask, 'transaction_frequency'] = transaction_frequency
            features.loc[mask, 'debit_ratio'] = debit_ratio
            features.loc[mask, 'credit_ratio'] = credit_ratio
            features.loc[mask, 'transfer_ratio'] = transfer_ratio
            features.loc[mask, 'payment_ratio'] = payment_ratio
            features.loc[mask, 'online_ratio'] = online_ratio
            features.loc[mask, 'mobile_ratio'] = mobile_ratio
            features.loc[mask, 'atm_ratio'] = atm_ratio
            features.loc[mask, 'branch_ratio'] = branch_ratio
            features.loc[mask, 'days_since_last_transaction'] = days_since_last
        
        # Convert to numeric
        numeric_cols = [
            'total_transactions', 'total_amount', 'avg_transaction_amount',
            'transaction_frequency', 'debit_ratio', 'credit_ratio',
            'transfer_ratio', 'payment_ratio', 'online_ratio', 'mobile_ratio',
            'atm_ratio', 'branch_ratio', 'days_since_last_transaction'
        ]
        
        for col in numeric_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        self.logger.info(f"Created {len(features.columns)} transaction features")
        return features
    
    def create_balance_features(
        self,
        customer_df: pd.DataFrame,
        balance_df: pd.DataFrame,
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Create balance-based features.
        
        Args:
            customer_df: Customer base data
            balance_df: Account balance history
            lookback_days: Number of days to look back for features
            
        Returns:
            DataFrame with balance features
        """
        self.logger.info(f"Creating balance features (lookback: {lookback_days} days)")
        
        features = customer_df[['customer_id']].copy()
        
        # Calculate cutoff date
        max_date = balance_df['date'].max()
        cutoff_date = max_date - timedelta(days=lookback_days)
        
        for customer_id in customer_df['customer_id']:
            customer_balances = balance_df[
                (balance_df['customer_id'] == customer_id) &
                (balance_df['date'] >= cutoff_date)
            ].sort_values('date')
            
            if len(customer_balances) == 0:
                # No balance data in lookback period
                features.loc[features['customer_id'] == customer_id, 'avg_balance'] = 0
                features.loc[features['customer_id'] == customer_id, 'min_balance'] = 0
                features.loc[features['customer_id'] == customer_id, 'max_balance'] = 0
                features.loc[features['customer_id'] == customer_id, 'balance_volatility'] = 0
                features.loc[features['customer_id'] == customer_id, 'balance_trend'] = 0
                features.loc[features['customer_id'] == customer_id, 'low_balance_days'] = 0
                features.loc[features['customer_id'] == customer_id, 'zero_balance_days'] = 0
                continue
            
            balances = customer_balances['balance'].values
            
            # Basic balance metrics
            avg_balance = balances.mean()
            min_balance = balances.min()
            max_balance = balances.max()
            balance_volatility = balances.std()
            
            # Balance trend (linear regression slope)
            if len(balances) > 1:
                x = np.arange(len(balances))
                balance_trend = np.polyfit(x, balances, 1)[0]
            else:
                balance_trend = 0
            
            # Low balance indicators
            low_balance_threshold = avg_balance * 0.1  # 10% of average balance
            low_balance_days = (balances < low_balance_threshold).sum()
            zero_balance_days = (balances == 0).sum()
            
            # Update features
            mask = features['customer_id'] == customer_id
            features.loc[mask, 'avg_balance'] = avg_balance
            features.loc[mask, 'min_balance'] = min_balance
            features.loc[mask, 'max_balance'] = max_balance
            features.loc[mask, 'balance_volatility'] = balance_volatility
            features.loc[mask, 'balance_trend'] = balance_trend
            features.loc[mask, 'low_balance_days'] = low_balance_days
            features.loc[mask, 'zero_balance_days'] = zero_balance_days
        
        # Convert to numeric
        numeric_cols = [
            'avg_balance', 'min_balance', 'max_balance', 'balance_volatility',
            'balance_trend', 'low_balance_days', 'zero_balance_days'
        ]
        
        for col in numeric_cols:
            features[col] = pd.to_numeric(features[col], errors='coerce')
        
        self.logger.info(f"Created {len(features.columns)} balance features")
        return features
    
    def create_risk_features(
        self,
        customer_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        balance_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create risk indicator features.
        
        Args:
            customer_df: Customer base data
            transaction_df: Transaction history
            balance_df: Account balance history
            
        Returns:
            DataFrame with risk features
        """
        self.logger.info("Creating risk indicator features")
        
        features = customer_df[['customer_id']].copy()
        
        for customer_id in customer_df['customer_id']:
            # Get customer data
            customer_transactions = transaction_df[
                transaction_df['customer_id'] == customer_id
            ]
            customer_balances = balance_df[
                balance_df['customer_id'] == customer_id
            ]
            
            # Risk indicators
            risk_score = 0
            
            # High transaction amounts relative to income
            customer_info = customer_df[customer_df['customer_id'] == customer_id].iloc[0]
            income = customer_info['income']
            
            if len(customer_transactions) > 0:
                max_transaction = customer_transactions['amount'].max()
                if max_transaction > income * 0.2:  # Transaction > 20% of income
                    risk_score += 1
            
            # Frequent low balances
            if len(customer_balances) > 0:
                low_balance_threshold = income * 0.05  # 5% of income
                low_balance_ratio = (customer_balances['balance'] < low_balance_threshold).mean()
                if low_balance_ratio > 0.5:  # Low balance > 50% of time
                    risk_score += 1
            
            # High debit-to-credit ratio
            if len(customer_transactions) > 0:
                debit_count = len(customer_transactions[
                    customer_transactions['transaction_type'] == 'debit'
                ])
                credit_count = len(customer_transactions[
                    customer_transactions['transaction_type'] == 'credit'
                ])
                
                if credit_count > 0:
                    debit_ratio = debit_count / (debit_count + credit_count)
                    if debit_ratio > 0.8:  # High debit ratio
                        risk_score += 1
            
            # Inactive account
            if len(customer_transactions) > 0:
                days_since_last_transaction = (
                    customer_transactions['transaction_date'].max() -
                    customer_transactions['transaction_date'].min()
                ).days
                
                if days_since_last_transaction > 90:  # Inactive for > 90 days
                    risk_score += 1
            
            # Update features
            mask = features['customer_id'] == customer_id
            features.loc[mask, 'risk_score'] = risk_score
            features.loc[mask, 'is_high_risk'] = int(risk_score >= 2)
        
        # Convert to numeric
        features['risk_score'] = pd.to_numeric(features['risk_score'], errors='coerce')
        features['is_high_risk'] = pd.to_numeric(features['is_high_risk'], errors='coerce')
        
        self.logger.info(f"Created {len(features.columns)} risk features")
        return features
    
    def create_all_features(
        self,
        datasets: Dict[str, pd.DataFrame],
        lookback_days: int = 90
    ) -> pd.DataFrame:
        """
        Create all features for the dataset.
        
        Args:
            datasets: Dictionary containing all datasets
            lookback_days: Number of days to look back for temporal features
            
        Returns:
            DataFrame with all features
        """
        self.logger.info("Creating comprehensive feature set")
        
        customer_df = datasets['customers']
        transaction_df = datasets['transactions']
        balance_df = datasets['balances']
        labels_df = datasets['labels']
        
        # Create different feature groups
        demographic_features = self.create_demographic_features(customer_df)
        transaction_features = self.create_transaction_features(
            customer_df, transaction_df, lookback_days
        )
        balance_features = self.create_balance_features(
            customer_df, balance_df, lookback_days
        )
        risk_features = self.create_risk_features(
            customer_df, transaction_df, balance_df
        )
        
        # Merge all features
        features = demographic_features.merge(
            transaction_features, on='customer_id', how='left'
        ).merge(
            balance_features, on='customer_id', how='left'
        ).merge(
            risk_features, on='customer_id', how='left'
        ).merge(
            labels_df, on='customer_id', how='left'
        )
        
        # Fill missing values (handle categorical columns properly)
        for col in features.columns:
            if features[col].dtype.name == 'category':
                # For categorical columns, fill with the first category
                features[col] = features[col].fillna(features[col].cat.categories[0])
            else:
                # For numeric columns, fill with 0
                features[col] = features[col].fillna(0)
        
        # Store feature names
        self.feature_names = [col for col in features.columns 
                             if col not in ['customer_id', 'churn', 'churn_probability']]
        
        self.logger.info(f"Created {len(self.feature_names)} features for {len(features)} customers")
        return features
    
    def save_features(self, features_df: pd.DataFrame, output_path: str) -> None:
        """
        Save features to file.
        
        Args:
            features_df: Features DataFrame
            output_path: Output file path
        """
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        features_df.to_parquet(output_path, index=False)
        self.logger.info(f"Saved features to {output_path}")
        
        # Save feature names
        feature_names_file = output_file.parent / f"{output_file.stem}_feature_names.txt"
        with open(feature_names_file, 'w') as f:
            for feature_name in self.feature_names:
                f.write(f"{feature_name}\n")
        
        self.logger.info(f"Saved feature names to {feature_names_file}")
    
    def load_features(self, input_path: str) -> pd.DataFrame:
        """
        Load features from file.
        
        Args:
            input_path: Input file path
            
        Returns:
            Features DataFrame
        """
        features_df = pd.read_parquet(input_path)
        
        # Load feature names
        input_file = Path(input_path)
        feature_names_file = input_file.parent / f"{input_file.stem}_feature_names.txt"
        
        if feature_names_file.exists():
            with open(feature_names_file, 'r') as f:
                self.feature_names = [line.strip() for line in f.readlines()]
        
        self.logger.info(f"Loaded features from {input_path}")
        return features_df
