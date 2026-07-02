"""
Data Loading module for Retail Sales Analytics
"""

import pandas as pd
import logging
from pathlib import Path
from src.config import Config

logger = logging.getLogger(__name__)

class DataLoader:
    """Load retail sales data from CSV"""
    
    @staticmethod
    def load_raw_data(filepath: str = None) -> pd.DataFrame:
        """
        Load raw sales data from CSV.
        
        Args:
            filepath: Path to CSV file (uses default if None)
             
        Returns:
            DataFrame with raw sales data
        """
        if filepath is None:
            filepath = Config.RAW_DATA_FILE
        
        try:
            logger.info(f"Loading data from {filepath}...")
            df = pd.read_csv(filepath)
            logger.info(f"✓ Loaded {len(df)} records with {len(df.columns)} columns")
            return df
        except FileNotFoundError:
            logger.error(f"Data file not found: {filepath}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    @staticmethod
    def save_processed_data(df: pd.DataFrame, filepath: str = None) -> bool:
        """Save processed data to CSV"""
        if filepath is None:
            filepath = Config.PROCESSED_DATA_FILE
        
        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(filepath, index=False)
            logger.info(f"✓ Saved {len(df)} records to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error saving data: {str(e)}")
            return False