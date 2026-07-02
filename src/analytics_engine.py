"""
Analytics Engine for Retail Sales Analytics
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class AnalyticsEngine:
    """Generate insights from sales data"""
    
    def __init__(self, df: pd.DataFrame):
        """Initialize with processed data"""
        self.df = df
        self.insights = {}
        logger.info("Analytics Engine initialized")
    
    def analyze_sales_by_category(self) -> dict:
        """Analyze sales performance by category"""
        if 'Category' not in self.df.columns:
            return {}
        
        category_analysis = self.df.groupby('Category').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Profit': 'sum',
            'Quantity': 'sum'
        }).round(2)
        
        self.insights['sales_by_category'] = category_analysis
        logger.info("✓ Category analysis completed")
        return category_analysis
    
    def analyze_monthly_trends(self) -> dict:
        """Analyze monthly sales trends"""
        if 'Month' not in self.df.columns:
            return {}
        
        monthly_trend = self.df.groupby('Month').agg({
            'Revenue': ['sum', 'mean', 'count'],
            'Profit': 'sum',
            'Quantity': 'sum'
        }).round(2)
        
        self.insights['monthly_trends'] = monthly_trend
        logger.info("✓ Monthly trend analysis completed")
        return monthly_trend
    
    def analyze_customer_segments(self) -> dict:
        """Segment customers by spending"""
        if 'Customer' not in self.df.columns:
            return {}
        
        customer_spend = self.df.groupby('Customer').agg({
            'Revenue': 'sum',
            'Quantity': 'sum',
            'Product': 'count'
        }).rename(columns={'Product': 'Purchases'}).round(2)
        
        # Categorize customers
        customer_spend['Segment'] = pd.cut(
            customer_spend['Revenue'],
            bins=[0, 1000, 5000, 100000],
            labels=['Low Value', 'Medium Value', 'High Value']
        )
        
        self.insights['customer_segments'] = customer_spend
        logger.info("✓ Customer segmentation completed")
        return customer_spend
    
    def analyze_profitability(self) -> dict:
        """Analyze profit metrics"""
        if 'Profit' not in self.df.columns:
            return {}
        
        profit_analysis = {
            'Total Revenue': self.df['Revenue'].sum(),
            'Total Profit': self.df['Profit'].sum(),
            'Avg Profit Margin': self.df['Profit_Margin'].mean(),
            'High Profit Products': self.df.nlargest(5, 'Profit')[['Product', 'Profit']].to_dict()
        }
        
        self.insights['profitability'] = profit_analysis
        logger.info("✓ Profitability analysis completed")
        return profit_analysis
    
    def get_summary_report(self) -> str:
        """Generate text summary report"""
        report = []
        report.append("=" * 60)
        report.append("RETAIL SALES ANALYTICS - SUMMARY REPORT")
        report.append("=" * 60)
        report.append("")
        
        report.append("KEY METRICS:")
        report.append(f"Total Records: {len(self.df)}")
        report.append(f"Total Revenue: PKR{self.df['Revenue'].sum():,.2f}")
        report.append(f"Total Profit: PKR{self.df['Profit'].sum():,.2f}")
        report.append(f"Avg Profit Margin: {self.df['Profit_Margin'].mean():.2f}%")
        report.append("")
        
        report.append("TOP 5 PRODUCTS BY REVENUE:")
        top_products = (
            self.df.groupby('Product')
            .agg(Revenue=('Revenue', 'sum'), Profit=('Profit', 'sum'))
            .reset_index()
            .nlargest(5, 'Revenue')
            )
        for idx, row in top_products.iterrows():
            report.append(f"  {row['Product']}: PKR{row['Revenue']:,.0f} (Profit: PKR{row['Profit']:,.0f})")
        report.append("")
        
        report.append("CATEGORY PERFORMANCE:")
        category_rev = self.df.groupby('Category')['Revenue'].sum().sort_values(ascending=False)
        for category, revenue in category_rev.items():
            report.append(f"  {category}: PKR{revenue:,.0f}")
        report.append("")
        
        report.append("=" * 60)
        
        return "\n".join(report)