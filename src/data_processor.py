"""
Data Processing module for Retail Sales Analytics
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class DataProcessor:
    """Process and clean retail sales data"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with DataFrame"""
        self.df = df.copy()
        self.original_count = len(df)
        logger.info(f"Processing {len(df)} records...")
    
    def clean_data(self):
        """Clean data: remove nulls, duplicates, fix types"""
        # Remove duplicates
        initial = len(self.df)
        self.df = self.df.drop_duplicates()
        removed = initial - len(self.df)
        if removed > 0:
            logger.info(f"Removed {removed} duplicate records")
        
        # Handle missing values
        self.df = self.df.dropna()
        
        # Fix data types
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        logger.info(f"✓ Data cleaned: {len(self.df)} records remain")
        return self
    
    def add_features(self):
        """Add derived features"""
        # Revenue feature
        if 'Quantity' in self.df.columns and 'Price' in self.df.columns:
            self.df['Revenue'] = self.df['Quantity'] * self.df['Price']
        
        # Profit feature
        if 'Revenue' in self.df.columns and 'Cost' in self.df.columns:
            self.df['Profit'] = self.df['Revenue'] - self.df['Cost']
            self.df['Profit_Margin'] = (self.df['Profit'] / self.df['Revenue'] * 100).round(2)
        
        # Month from date
        if 'Date' in self.df.columns:
            self.df['Month'] = self.df['Date'].dt.to_period('M')
            self.df['Year'] = self.df['Date'].dt.year
        
        logger.info("✓ Features added")
        return self
    
    def validate_data(self) -> bool:
        """Validate data integrity"""
        required_cols = ['Product', 'Category', 'Quantity', 'Price']
        missing = [col for col in required_cols if col not in self.df.columns]
        
        if missing:
            logger.warning(f"Missing columns: {missing}")
            return False
        
        # Check for negative values
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if (self.df[col] < 0).any():
                logger.warning(f"Found negative values in {col}")
        
        logger.info("✓ Data validation passed")
        return True
    
    def get_processed_data(self) -> pd.DataFrame:
        """Get processed DataFrame"""
        return self.df.copy()
