import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from matplotlib.ticker import FuncFormatter


# ---------------------------------------------------
# LOGGING CONFIGURATION
# ---------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------
# MATPLOTLIB SETTINGS
# ---------------------------------------------------

plt.rcParams.update({
    "figure.figsize": (9, 4.5),
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------------------------------------------------
# HELPER FUNCTION
# ---------------------------------------------------

def currency_k(x, pos):

    if x >= 1_000_000:
        return f"{x/1_000_000:.1f}M"

    elif x >= 1_000:
        return f"{x/1_000:.0f}K"

    else:
        return f"{x:.0f}"


# ---------------------------------------------------
# MAIN CLASS
# ---------------------------------------------------

class RetailAnalyticsDashboard:

    # ---------------------------------------------------
    # INIT
    # ---------------------------------------------------

    def __init__(self, filepath):

        logger.info("Loading Dataset")

        self.df = pd.read_csv(filepath)

        logger.info(f"Dataset Loaded | Shape: {self.df.shape}")

        self.df["order_date"] = pd.to_datetime(
            self.df["order_date"],
            errors="coerce"
        )

        invalid_dates = self.df["order_date"].isna().sum()

        if invalid_dates > 0:
            logger.warning(f"Invalid Dates Found: {invalid_dates}")

        self.df.dropna(subset=["order_date"], inplace=True)

        self.df.set_index("order_date", inplace=True)

        logger.info("Date Conversion Successful")

        # Better pipeline flow
        self.data_cleaning()

        self.feature_engineering()

    # ---------------------------------------------------
    # BASIC EDA
    # ---------------------------------------------------

    def basic_eda(self):

        logger.info("Running Basic EDA")

        print("\nTop 5 Rows:\n", self.df.head())

        print("\nDataset Shape:", self.df.shape)

        print("\nDuplicate Rows:", self.df.duplicated().sum())

        print("\nMissing Values:\n", self.df.isna().sum())

        print("\nData Types:\n", self.df.dtypes)

        print("\nRevenue Summary:\n", self.df["Revenue"].describe())

    # ---------------------------------------------------
    # DATA CLEANING
    # ---------------------------------------------------

    def data_cleaning(self):

        logger.info("Starting Data Cleaning")

        before_rows = len(self.df)

        # Remove Negative Quantity
        self.df = self.df[self.df["quantity"] >= 0]

        logger.info("Negative Quantities Removed")

        # Remove Duplicates
        duplicates = self.df.duplicated().sum()

        self.df.drop_duplicates(inplace=True)

        logger.info(f"Duplicate Rows Removed: {duplicates}")

        # Fill Missing Discount
        missing_discount = self.df["discount"].isna().sum()

        if missing_discount > 0:

            logger.info(f"Missing Discounts Found: {missing_discount}")

            self.df["discount"] = self.df["discount"].fillna(0)

            logger.info("Missing Discounts Filled")

        else:

            logger.info("No Missing Discounts Found")
           

        after_rows = len(self.df)

        logger.info(
            f"Cleaning Completed | Before: {before_rows} | After: {after_rows}"
        )

    # ---------------------------------------------------
    # FEATURE ENGINEERING
    # ---------------------------------------------------

    def feature_engineering(self):

        logger.info("Running Feature Engineering")

        # Revenue
        self.df["Revenue"] = (
            self.df["price"] *
            self.df["quantity"] *
            (1 - self.df["discount"])
        )

        # Day Features
        self.df["Day_Name"] = self.df.index.day_name()

        self.df["Month"] = self.df.index.month_name()

        self.df["Is_Weekend"] = np.where(
            self.df.index.dayofweek >= 5,
            1,
            0
        )

        logger.info("Feature Engineering Completed")

    # ---------------------------------------------------
    # KPI DASHBOARD
    # ---------------------------------------------------

    @property
    def generate_kpi_dashboard(self):

        logger.info("Generating KPI Dashboard")

        total_revenue = self.df["Revenue"].sum()

        total_orders = self.df["order_id"].nunique()

        total_customers = self.df["customer_id"].nunique()

        avg_order_value = (
            total_revenue / total_orders
            if total_orders != 0
            else 0
        )

        best_region = (
            self.df.groupby("region")["Revenue"]
            .sum()
            .idxmax()
        )

        best_category = (
            self.df.groupby("product_category")["Revenue"]
            .sum()
            .idxmax()
        )

        return {

            "Total Revenue": round(total_revenue, 2),

            "Total Orders": total_orders,

            "Total Customers": total_customers,

            "Average Order Value": round(avg_order_value, 2),

            "Best Region": best_region,

            "Best Product Category": best_category
        }

    # ---------------------------------------------------
    # TIME SERIES ANALYSIS
    # ---------------------------------------------------

    def time_series_analysis(self):

        logger.info("Running Time Series Analysis")

        daily = self.df["Revenue"].resample("D").sum()

        rolling = daily.rolling(7).mean()

        monthly = daily.resample("ME").sum()

        logger.info("Time Series Analysis Completed")

        return daily, rolling, monthly

    # ---------------------------------------------------
    # TREND DIRECTION
    # ---------------------------------------------------

    def trend_direction(self):

        daily = self.df["Revenue"].resample("D").sum()

        diff = daily.diff().mean()

        if diff > 0:
            trend = "Increasing"

        elif diff < 0:
            trend = "Decreasing"

        else:
            trend = "Stable"

        logger.info(f"Trend Direction: {trend}")

        return trend

    # ---------------------------------------------------
    # REVENUE TREND PLOT
    # ---------------------------------------------------

    def revenue_trend_plot(self):

        logger.info("Generating Revenue Trend Plot")

        daily = self.df["Revenue"].resample("D").sum()

        rolling = daily.rolling(7).mean()

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            daily.index,
            daily,
            alpha=0.4,
            linewidth=1,
            label="Daily Revenue"
        )

        ax.plot(
            rolling.index,
            rolling,
            linewidth=2.5,
            label="7-Day Rolling Avg"
        )

        ax.set_title(
            "Revenue Trend Analysis",
            fontweight="bold"
        )

        ax.set_xlabel("Date")

        ax.set_ylabel("Revenue")

        ax.legend(frameon=False)

        ax.yaxis.set_major_formatter(
            FuncFormatter(currency_k)
        )

        ax.xaxis.set_major_locator(
            mdates.AutoDateLocator()
        )

        ax.xaxis.set_major_formatter(
            mdates.DateFormatter("%b %d")
        )

        fig.autofmt_xdate()

        plt.tight_layout()

        plt.show()

        logger.info("Revenue Trend Plot Displayed")

    # ---------------------------------------------------
    # REGION REVENUE PLOT
    # ---------------------------------------------------

    def region_revenue_plot(self):

        logger.info("Generating Region Revenue Plot")

        region_rev = (
            self.df.groupby("region")["Revenue"]
            .sum()
            .sort_values()
        )

        fig, ax = plt.subplots(figsize=(7, 4))

        ax.barh(
            region_rev.index,
            region_rev.values
        )

        ax.set_title(
            "Revenue by Region",
            fontweight="bold"
        )

        ax.set_xlabel("Revenue")

        ax.xaxis.set_major_formatter(
            FuncFormatter(currency_k)
        )

        plt.tight_layout()

        plt.show()

        logger.info("Region Revenue Plot Displayed")

    # ---------------------------------------------------
    # TOP PRODUCT CATEGORIES
    # ---------------------------------------------------

    def top_categories_plot(self):

        logger.info("Generating Product Category Plot")

        category_rev = (
            self.df.groupby("product_category")["Revenue"]
            .sum()
            .sort_values(ascending=False)
        )

        fig, ax = plt.subplots(figsize=(8, 4))

        ax.bar(
            category_rev.index,
            category_rev.values
        )

        ax.set_title(
            "Top Product Categories",
            fontweight="bold"
        )

        ax.set_ylabel("Revenue")

        ax.yaxis.set_major_formatter(
            FuncFormatter(currency_k)
        )

        plt.tight_layout()

        plt.show()

        logger.info("Category Plot Displayed")

    # ---------------------------------------------------
    # CUSTOMER ANALYSIS
    # ---------------------------------------------------

    def customer_analysis(self):

        logger.info("Running Customer Analysis")

        customer = self.df.groupby("customer_id").agg(

            total_spending=("Revenue", "sum"),

            total_orders=("order_id", "count"),

            avg_order_value=("Revenue", "mean")
        )

        customer["Customer_Type"] = np.where(
            customer["total_orders"] > 1,
            "Repeat",
            "New"
        )

        self.customer_df = customer

        logger.info("Customer Analysis Completed")

        print("\nCustomer Analysis:\n", customer.head())

    # ---------------------------------------------------
    # EXPORT CLEAN DATA
    # ---------------------------------------------------

    def export_clean_data(self):

        self.df.to_csv("cleaned_retail_data.csv")

        logger.info("Cleaned Dataset Exported Successfully")


# ---------------------------------------------------
# MAIN
# ---------------------------------------------------

if __name__ == "__main__":

    logger.info("Project Started")

    dashboard = RetailAnalyticsDashboard("advanced_retail_dataset_2024.csv")

    # EDA
    dashboard.basic_eda()

    # Cleaning
    dashboard.data_cleaning()

    # KPI
    print("\n=== KPI DASHBOARD ===")

    kpis = dashboard.generate_kpi_dashboard

    for k, v in kpis.items():
        print(f"{k}: {v}")

    # Trend
    print("\nTrend Direction:", dashboard.trend_direction())

    # Visualizations
    dashboard.revenue_trend_plot()

    dashboard.region_revenue_plot()

    dashboard.top_categories_plot()

    # Customer Analysis
    dashboard.customer_analysis()

    # Export Clean Data
    dashboard.export_clean_data()

    logger.info("Project Completed Successfully")