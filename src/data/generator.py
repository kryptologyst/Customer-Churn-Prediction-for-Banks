"""
Data generation and processing for customer churn prediction.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from pathlib import Path

from utils.reproducibility import set_seed


class CustomerDataGenerator:
    """
    Generate synthetic customer data for churn prediction.
    
    This class creates realistic customer data with temporal patterns,
    demographic distributions, and behavioral features that are
    commonly used in banking churn prediction.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the data generator.
        
        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        set_seed(seed)
        self.logger = logging.getLogger(__name__)
    
    def generate_customer_base(self, n_customers: int = 10000) -> pd.DataFrame:
        """
        Generate base customer demographic data.
        
        Args:
            n_customers: Number of customers to generate
            
        Returns:
            DataFrame with customer base information
        """
        self.logger.info(f"Generating base data for {n_customers} customers")
        
        # Customer IDs
        customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
        
        # Demographics
        ages = np.random.normal(45, 15, n_customers).astype(int)
        ages = np.clip(ages, 18, 80)  # Reasonable age range
        
        genders = np.random.choice(['M', 'F'], n_customers, p=[0.48, 0.52])
        
        # Income (log-normal distribution)
        incomes = np.random.lognormal(10.5, 0.8, n_customers)
        incomes = np.clip(incomes, 20000, 500000)  # Reasonable income range
        
        # Education levels
        education_levels = np.random.choice(
            ['High School', 'Bachelor', 'Master', 'PhD'],
            n_customers,
            p=[0.35, 0.40, 0.20, 0.05]
        )
        
        # Geographic regions
        regions = np.random.choice(
            ['Urban', 'Suburban', 'Rural'],
            n_customers,
            p=[0.50, 0.35, 0.15]
        )
        
        # Account tenure (years with bank)
        tenure_years = np.random.exponential(5, n_customers)
        tenure_years = np.clip(tenure_years, 0.1, 30)  # 1 month to 30 years
        
        # Account types
        account_types = np.random.choice(
            ['Basic', 'Premium', 'Business', 'Student'],
            n_customers,
            p=[0.50, 0.30, 0.15, 0.05]
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'customer_id': customer_ids,
            'age': ages,
            'gender': genders,
            'income': incomes,
            'education': education_levels,
            'region': regions,
            'tenure_years': tenure_years,
            'account_type': account_types,
        })
        
        self.logger.info(f"Generated base data for {len(df)} customers")
        return df
    
    def generate_transaction_history(
        self,
        customer_df: pd.DataFrame,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31"
    ) -> pd.DataFrame:
        """
        Generate transaction history for customers.
        
        Args:
            customer_df: Customer base data
            start_date: Start date for transactions
            end_date: End date for transactions
            
        Returns:
            DataFrame with transaction history
        """
        self.logger.info("Generating transaction history")
        
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        
        transactions = []
        
        for _, customer in customer_df.iterrows():
            customer_id = customer['customer_id']
            income = customer['income']
            account_type = customer['account_type']
            
            # Base transaction frequency (transactions per month)
            if account_type == 'Business':
                base_freq = np.random.poisson(25)  # Business customers transact more
            elif account_type == 'Premium':
                base_freq = np.random.poisson(15)
            else:
                base_freq = np.random.poisson(8)
            
            # Generate transactions over time
            current_date = start_dt
            while current_date <= end_dt:
                # Transaction frequency varies by month
                monthly_freq = max(1, int(base_freq * np.random.uniform(0.5, 1.5)))
                
                for _ in range(monthly_freq):
                    # Transaction amount (log-normal, correlated with income)
                    base_amount = income * np.random.uniform(0.001, 0.05)
                    amount = np.random.lognormal(np.log(base_amount), 0.5)
                    amount = max(1, min(amount, income * 0.1))  # Reasonable limits
                    
                    # Transaction type
                    transaction_type = np.random.choice(
                        ['debit', 'credit', 'transfer', 'payment'],
                        p=[0.40, 0.30, 0.20, 0.10]
                    )
                    
                    # Channel
                    channel = np.random.choice(
                        ['online', 'mobile', 'atm', 'branch'],
                        p=[0.50, 0.30, 0.15, 0.05]
                    )
                    
                    transactions.append({
                        'customer_id': customer_id,
                        'transaction_date': current_date,
                        'amount': amount,
                        'transaction_type': transaction_type,
                        'channel': channel,
                    })
                
                # Move to next month
                current_date += timedelta(days=30)
        
        transaction_df = pd.DataFrame(transactions)
        transaction_df['transaction_date'] = pd.to_datetime(transaction_df['transaction_date'])
        
        self.logger.info(f"Generated {len(transaction_df)} transactions")
        return transaction_df
    
    def generate_account_balances(
        self,
        customer_df: pd.DataFrame,
        transaction_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Generate account balance history.
        
        Args:
            customer_df: Customer base data
            transaction_df: Transaction history
            
        Returns:
            DataFrame with account balance history
        """
        self.logger.info("Generating account balance history")
        
        balances = []
        
        for _, customer in customer_df.iterrows():
            customer_id = customer['customer_id']
            income = customer['income']
            
            # Initial balance (correlated with income)
            initial_balance = income * np.random.uniform(0.1, 0.5)
            
            # Get customer transactions
            customer_transactions = transaction_df[
                transaction_df['customer_id'] == customer_id
            ].sort_values('transaction_date')
            
            current_balance = initial_balance
            
            for _, transaction in customer_transactions.iterrows():
                amount = transaction['amount']
                transaction_type = transaction['transaction_type']
                
                # Update balance based on transaction type
                if transaction_type == 'debit':
                    current_balance -= amount
                elif transaction_type == 'credit':
                    current_balance += amount
                elif transaction_type == 'transfer':
                    # Assume 50% chance of incoming/outgoing
                    if np.random.random() < 0.5:
                        current_balance += amount
                    else:
                        current_balance -= amount
                else:  # payment
                    current_balance -= amount
                
                # Ensure non-negative balance (overdraft protection)
                current_balance = max(0, current_balance)
                
                balances.append({
                    'customer_id': customer_id,
                    'date': transaction['transaction_date'],
                    'balance': current_balance,
                })
        
        balance_df = pd.DataFrame(balances)
        balance_df['date'] = pd.to_datetime(balance_df['date'])
        
        self.logger.info(f"Generated {len(balance_df)} balance records")
        return balance_df
    
    def generate_churn_labels(
        self,
        customer_df: pd.DataFrame,
        transaction_df: pd.DataFrame,
        balance_df: pd.DataFrame,
        churn_rate: float = 0.15
    ) -> pd.DataFrame:
        """
        Generate churn labels based on customer behavior patterns.
        
        Args:
            customer_df: Customer base data
            transaction_df: Transaction history
            balance_df: Account balance history
            churn_rate: Overall churn rate
            
        Returns:
            DataFrame with churn labels
        """
        self.logger.info(f"Generating churn labels with {churn_rate:.1%} churn rate")
        
        labels = []
        
        for _, customer in customer_df.iterrows():
            customer_id = customer['customer_id']
            income = customer['income']
            tenure_years = customer['tenure_years']
            account_type = customer['account_type']
            
            # Get customer data
            customer_transactions = transaction_df[
                transaction_df['customer_id'] == customer_id
            ]
            customer_balances = balance_df[
                balance_df['customer_id'] == customer_id
            ]
            
            # Calculate churn probability based on features
            churn_prob = 0.0
            
            # Lower income customers more likely to churn
            if income < 30000:
                churn_prob += 0.3
            elif income < 50000:
                churn_prob += 0.15
            
            # Newer customers more likely to churn
            if tenure_years < 1:
                churn_prob += 0.4
            elif tenure_years < 2:
                churn_prob += 0.2
            
            # Account type affects churn
            if account_type == 'Student':
                churn_prob += 0.2  # Students often churn after graduation
            
            # Transaction patterns
            if len(customer_transactions) > 0:
                recent_transactions = customer_transactions[
                    customer_transactions['transaction_date'] >= 
                    customer_transactions['transaction_date'].max() - timedelta(days=30)
                ]
                
                # Low activity in last 30 days increases churn risk
                if len(recent_transactions) < 2:
                    churn_prob += 0.3
                
                # High debit-to-credit ratio increases churn risk
                debit_count = len(customer_transactions[
                    customer_transactions['transaction_type'] == 'debit'
                ])
                credit_count = len(customer_transactions[
                    customer_transactions['transaction_type'] == 'credit'
                ])
                
                if credit_count > 0:
                    debit_ratio = debit_count / (debit_count + credit_count)
                    if debit_ratio > 0.8:
                        churn_prob += 0.2
            
            # Balance patterns
            if len(customer_balances) > 0:
                recent_balances = customer_balances[
                    customer_balances['date'] >= 
                    customer_balances['date'].max() - timedelta(days=30)
                ]
                
                # Low or declining balance increases churn risk
                if len(recent_balances) > 0:
                    avg_recent_balance = recent_balances['balance'].mean()
                    if avg_recent_balance < income * 0.05:  # Less than 5% of income
                        churn_prob += 0.25
            
            # Add some randomness
            churn_prob += np.random.normal(0, 0.1)
            churn_prob = np.clip(churn_prob, 0, 1)
            
            # Determine churn label
            churn_label = 1 if churn_prob > (1 - churn_rate) else 0
            
            labels.append({
                'customer_id': customer_id,
                'churn': churn_label,
                'churn_probability': churn_prob,
            })
        
        labels_df = pd.DataFrame(labels)
        actual_churn_rate = labels_df['churn'].mean()
        
        self.logger.info(f"Generated churn labels - actual rate: {actual_churn_rate:.1%}")
        return labels_df
    
    def generate_complete_dataset(
        self,
        n_customers: int = 10000,
        start_date: str = "2020-01-01",
        end_date: str = "2023-12-31",
        churn_rate: float = 0.15
    ) -> Dict[str, pd.DataFrame]:
        """
        Generate complete customer dataset.
        
        Args:
            n_customers: Number of customers
            start_date: Start date for data
            end_date: End date for data
            churn_rate: Target churn rate
            
        Returns:
            Dictionary containing all generated datasets
        """
        self.logger.info("Generating complete customer dataset")
        
        # Generate base customer data
        customer_df = self.generate_customer_base(n_customers)
        
        # Generate transaction history
        transaction_df = self.generate_transaction_history(
            customer_df, start_date, end_date
        )
        
        # Generate account balances
        balance_df = self.generate_account_balances(customer_df, transaction_df)
        
        # Generate churn labels
        labels_df = self.generate_churn_labels(
            customer_df, transaction_df, balance_df, churn_rate
        )
        
        return {
            'customers': customer_df,
            'transactions': transaction_df,
            'balances': balance_df,
            'labels': labels_df
        }
    
    def save_datasets(
        self,
        datasets: Dict[str, pd.DataFrame],
        output_dir: str = "data/raw"
    ) -> None:
        """
        Save generated datasets to files.
        
        Args:
            datasets: Dictionary of datasets
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for name, df in datasets.items():
            file_path = output_path / f"{name}.parquet"
            df.to_parquet(file_path, index=False)
            self.logger.info(f"Saved {name} dataset to {file_path}")
    
    def load_datasets(self, data_dir: str = "data/raw") -> Dict[str, pd.DataFrame]:
        """
        Load datasets from files.
        
        Args:
            data_dir: Data directory
            
        Returns:
            Dictionary of loaded datasets
        """
        data_path = Path(data_dir)
        datasets = {}
        
        for file_path in data_path.glob("*.parquet"):
            name = file_path.stem
            datasets[name] = pd.read_parquet(file_path)
            self.logger.info(f"Loaded {name} dataset from {file_path}")
        
        return datasets
