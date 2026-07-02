
"""
Configuration module for Retail Sales Analytics Dashboard
"""

from pathlib import Path
from datetime import datetime

class Config:
    """Application configuration"""
    
    # Paths
    PROJECT_ROOT = Path(__file__).parent.parent
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    OUTPUT_DIR = PROJECT_ROOT / "output"
    VISUALIZATIONS_DIR = OUTPUT_DIR / "visualizations"
    REPORTS_DIR = OUTPUT_DIR / "reports"
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    # Data Configuration
    RAW_DATA_FILE = RAW_DATA_DIR / "retail_sales_data.csv"
    PROCESSED_DATA_FILE = PROCESSED_DATA_DIR / "processed_sales.csv"
    
    # Analysis Settings
    MIN_SALES_THRESHOLD = 100
    HIGH_VALUE_CUSTOMER_SPEND = 5000
    PROFIT_MARGIN_TARGET = 0.30
    
    # Visualization
    PLOT_DPI = 100
    PLOT_STYLE = "seaborn-v0_8-darkgrid"
    COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    # Logging
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    
    @staticmethod
    def create_directories():
        """Create all necessary directories"""
        directories = [
            Config.RAW_DATA_DIR,
            Config.PROCESSED_DATA_DIR,
            Config.VISUALIZATIONS_DIR,
            Config.REPORTS_DIR,
            Config.LOGS_DIR
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)