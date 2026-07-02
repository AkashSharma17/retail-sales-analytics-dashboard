"""
Utility functions for Retail Sales Analytics Dashboard
"""

import os
import logging
import datetime
from pathlib import Path
from src.config import Config


def setup_logger():
    """Setup application logger — file + terminal dono mein"""
    Config.create_directories()

    logging.basicConfig(
        level=logging.INFO,
        format=Config.LOG_FORMAT,
        datefmt=Config.LOG_DATE_FORMAT,
        handlers=[
            logging.FileHandler(Config.LOGS_DIR / "app.log"),  
            logging.StreamHandler()                            
        ]
    )
    return logging.getLogger(__name__)


def format_currency(value):
    """Format value as Pakistani Rupees"""
    if value >= 10_000_000:
        return f'Rs.{value/10_000_000:.1f}Cr'
    elif value >= 100_000:
        return f'Rs.{value/100_000:.1f}L'
    elif value >= 1000:
        return f'Rs.{value/1000:.0f}K'
    else:
        return f'Rs.{value:.0f}'


def create_directory(path):
    """Create directory if it doesn't exist"""
    os.makedirs(path, exist_ok=True)


def get_current_timestamp():
    """Return current timestamp as string"""
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")