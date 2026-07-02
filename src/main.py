
"""
Main orchestrator for Retail Sales Analytics Dashboard
"""

import logging
from src.config import Config
from src.data_loader import DataLoader
from src.data_processor import DataProcessor
from src.analytics_engine import AnalyticsEngine
from src.visualizer import Visualizer

logger = logging.getLogger(__name__)

class RetailAnalyticsPipeline:
    """Complete retail analytics pipeline"""
    
    def __init__(self):
        """Initialize pipeline"""
        Config.create_directories()
        logger.info("=" * 60)
        logger.info("RETAIL SALES ANALYTICS DASHBOARD")
        logger.info("=" * 60)
    
    def run(self):
        """Execute complete pipeline"""
        try:
            # Step 1: Load Data
            logger.info("\n[STEP 1] Loading Data...")
            df = DataLoader.load_raw_data()
            
            # Step 2: Process Data
            logger.info("\n[STEP 2] Processing Data...")
            processor = DataProcessor(df)
            df_clean = processor.clean_data().add_features().get_processed_data()
            
            if not processor.validate_data():
                logger.warning("Data validation failed but continuing...")
            
            # Save processed data
            DataLoader.save_processed_data(df_clean)
            
            # Step 3: Analyze Data
            logger.info("\n[STEP 3] Analyzing Data...")
            engine = AnalyticsEngine(df_clean)
            engine.analyze_sales_by_category()
            engine.analyze_monthly_trends()
            engine.analyze_customer_segments()
            engine.analyze_profitability()
            
            # Step 4: Create Visualizations
            logger.info("\n[STEP 4] Creating Visualizations...")
            viz = Visualizer(df_clean)
            viz.plot_sales_by_category()
            viz.plot_monthly_trends()
            viz.plot_customer_analysis()
            viz.plot_profit_analysis()
            
            # Step 5: Generate Report
            logger.info("\n[STEP 5] Generating Report...")
            report = engine.get_summary_report()
            report_file = Config.REPORTS_DIR / "analysis_report.txt"
            with open(report_file, 'w') as f:
                f.write(report)
            logger.info(f"✓ Report saved: {report_file}")
            
            # Print summary
            print("\n" + report)
            
            logger.info("\n" + "=" * 60)
            logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
            return False


def main():
    """Entry point"""
    pipeline = RetailAnalyticsPipeline()
    success = pipeline.run()
    return success


if __name__ == "__main__":
    main()
