"""
Visualization module for Retail Sales Analytics
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import FuncFormatter
import seaborn as sns
import logging
from src.config import Config

logger = logging.getLogger(__name__)


def format_pkr(value, pos=None):
    """Format numbers as PKR with K/L/Cr suffix"""
    if value >= 10_000_000:
        return f'Rs.{value/10_000_000:.1f}Cr'
    elif value >= 100_000:
        return f'Rs.{value/100_000:.1f}L'
    elif value >= 1000:
        return f'Rs.{value/1000:.0f}K'
    else:
        return f'Rs.{value:.0f}'


class Visualizer:
    """Create visualizations for sales data"""

    def __init__(self, df):
        """Initialize visualizer"""
        self.df = df
        sns.set_style("whitegrid")

        # Font settings AFTER sns.set_style to prevent reset
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        plt.rcParams['figure.dpi'] = Config.PLOT_DPI

        logger.info("Visualizer initialized")

    def plot_sales_by_category(self):
        """Create category sales chart"""
        if 'Category' not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        category_sales = self.df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)

        category_sales.plot(kind='bar', ax=ax, color=Config.COLORS[0], alpha=0.8)
        ax.set_title('Total Revenue by Category', fontsize=14, fontweight='bold')
        ax.set_xlabel('Category', fontsize=12)
        ax.set_ylabel('Revenue (PKR)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        ax.yaxis.set_major_formatter(FuncFormatter(format_pkr))

        # Add value labels
        for i, v in enumerate(category_sales.values):
            ax.text(i, v + 1000, format_pkr(v), ha='center', va='bottom', fontsize=10)

        plt.tight_layout()
        filepath = Config.VISUALIZATIONS_DIR / "sales_by_category.png"
        plt.savefig(filepath, dpi=Config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved: {filepath}")
        plt.close()

    def plot_monthly_trends(self):
        """Create monthly trend chart"""
        if 'Month' not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        monthly_data = self.df.groupby('Month')['Revenue'].sum()

        ax.plot(range(len(monthly_data)), monthly_data.values, marker='o', linewidth=2,
                markersize=8, color=Config.COLORS[1])
        ax.fill_between(range(len(monthly_data)), monthly_data.values, alpha=0.3, color=Config.COLORS[1])

        ax.set_title('Monthly Revenue Trends', fontsize=14, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Revenue (PKR)', fontsize=12)
        ax.set_xticks(range(len(monthly_data)))
        ax.set_xticklabels([str(m) for m in monthly_data.index], rotation=45)
        ax.yaxis.set_major_formatter(FuncFormatter(format_pkr))
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = Config.VISUALIZATIONS_DIR / "monthly_trends.png"
        plt.savefig(filepath, dpi=Config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved: {filepath}")
        plt.close()

    def plot_customer_analysis(self):
        """Create customer segment chart"""
        if 'Customer' not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        customer_data = self.df.groupby('Customer')['Revenue'].sum().sort_values(ascending=False).head(10)

        customer_data.plot(kind='barh', ax=ax, color=Config.COLORS[2], alpha=0.8)
        ax.set_title('Top 10 Customers by Revenue', fontsize=14, fontweight='bold')
        ax.set_xlabel('Revenue (PKR)', fontsize=12)
        ax.set_ylabel('Customer', fontsize=12)
        ax.xaxis.set_major_formatter(FuncFormatter(format_pkr))

        # Add value labels
        for i, v in enumerate(customer_data.values):
            ax.text(v + 100, i, format_pkr(v), va='center', fontsize=10)

        plt.tight_layout()
        filepath = Config.VISUALIZATIONS_DIR / "customer_analysis.png"
        plt.savefig(filepath, dpi=Config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved: {filepath}")
        plt.close()

    def plot_profit_analysis(self):
        """Create profit distribution chart"""
        if 'Profit_Margin' not in self.df.columns:
            return

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.hist(self.df['Profit_Margin'], bins=30, color=Config.COLORS[3], alpha=0.7, edgecolor='black')
        ax.set_title('Profit Margin Distribution', fontsize=14, fontweight='bold')
        ax.set_xlabel('Profit Margin (%)', fontsize=12)
        ax.set_ylabel('Number of Products', fontsize=12)
        ax.axvline(self.df['Profit_Margin'].mean(), color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {self.df['Profit_Margin'].mean():.2f}%")
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        filepath = Config.VISUALIZATIONS_DIR / "profit_analysis.png"
        plt.savefig(filepath, dpi=Config.PLOT_DPI, bbox_inches='tight')
        logger.info(f"✓ Saved: {filepath}")
        plt.close()