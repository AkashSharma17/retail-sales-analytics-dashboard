
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib.ticker import FuncFormatter

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


def currency_k(x, pos):
    if x >= 1e6:
        return f'{x/1e6:.1f}M'
    elif x >= 1e3:
        return f'{x/1e3:.0f}K'
    else:
        return f'{x:.0f}'


np.random.seed(2026)


class RetailAnalyticsDashboard:

    def __init__(self, data: dict):
        self.df = pd.DataFrame(data)
        self.df["Date"] = pd.to_datetime(self.df["Date"], errors="coerce")
        self.df.dropna(subset=["Date"], inplace=True)
        self.df.set_index("Date", inplace=True)
        self.feature_engineering()

    # -------------------------
    # EDA
    # -------------------------

    def basic_eda(self):
        print("\nTop 5 Rows:\n", self.df.head())
        print("\nDuplicate Rows:", self.df.duplicated().sum())
        print("\nMissing Values:\n", self.df.isna().sum())
        print("\nData Types:\n", self.df.dtypes)
        print("\nUnits Sold Summary:\n", self.df["Units_Sold"].describe())
        print("\nPrice Summary:\n", self.df["Price"].describe())
        print("\nCost Summary:\n", self.df["Cost"].describe())

    def professional_eda(self):
        result = {}
        result["Top 5 Rows"] = self.df.head()
        result["Shape"] = self.df.shape
        result["Duplicate Rows"] = self.df.duplicated().sum()
        result["Missing Values"] = self.df.isna().sum()
        result["Data Types"] = self.df.dtypes

        numeric_cols = ["Units_Sold", "Price", "Cost", "Revenue", "Profit", "Profit_Margin"]
        existing_cols = [c for c in numeric_cols if c in self.df.columns]
        result["Numeric Summary"] = self.df[existing_cols].describe()

        if "Store" in self.df.columns:
            result["Store Revenue Summary"] = (
                self.df.groupby("Store")["Revenue"].agg(["max", "min", "mean"])
            )

        return result

    # -------------------------
    # DATA CLEANING
    # -------------------------

    def data_cleaning(self, drop_invalid_price=False):
        self.df = self.df[self.df["Units_Sold"] >= 0]
        self.df["Invalid_Price"] = np.where(self.df["Price"] < self.df["Cost"], 1, 0)

        if drop_invalid_price:
            self.df = self.df[self.df["Invalid_Price"] == 0]

        self.df.drop_duplicates(inplace=True)

    # -------------------------
    # FEATURE ENGINEERING
    # -------------------------

    def feature_engineering(self, high_sales_threshold=6000):
        self.df["Revenue"] = self.df["Units_Sold"] * self.df["Price"]
        self.df["Total_Cost"] = self.df["Units_Sold"] * self.df["Cost"]
        self.df["Profit"] = self.df["Revenue"] - self.df["Total_Cost"]

        self.df["Profit_Margin"] = np.where(
            self.df["Revenue"] == 0,
            0,
            self.df["Profit"] / self.df["Revenue"] * 100
        )

        self.df["Day_Name"] = self.df.index.day_name()
        self.df["Day_Number"] = self.df.index.dayofweek
        self.df["Is_Weekend"] = np.where(self.df["Day_Number"] >= 5, 1, 0)
        self.df["High_Sales"] = np.where(self.df["Revenue"] > high_sales_threshold, 1, 0)
        self.df["Loss_Flag"] = np.where(self.df["Profit"] < 0, 1, 0)

    # -------------------------
    # KPI DASHBOARD
    # -------------------------

    @property
    def generate_kpi_dashboard(self):

        total_revenue = self.df["Revenue"].sum()
        total_profit = self.df["Profit"].sum()
        margin = (total_profit / total_revenue * 100) if total_revenue != 0 else 0

        store_rev = self.df.groupby("Store", as_index=False)["Revenue"].sum()
        product_rev = self.df.groupby("Product", as_index=False)["Revenue"].sum()
        region_rev = self.df.groupby("Region", as_index=False)["Revenue"].sum()

        avg_daily_revenue = self.df["Revenue"].resample("D").sum().mean()

        best_store = store_rev.loc[store_rev["Revenue"].idxmax()] if not store_rev.empty else None
        worst_store = store_rev.loc[store_rev["Revenue"].idxmin()] if not store_rev.empty else None

        best_product = product_rev.loc[product_rev["Revenue"].idxmax()] if not product_rev.empty else None
        worst_product = product_rev.loc[product_rev["Revenue"].idxmin()] if not product_rev.empty else None

        best_region = region_rev.loc[region_rev["Revenue"].idxmax()] if not region_rev.empty else None
        worst_region = region_rev.loc[region_rev["Revenue"].idxmin()] if not region_rev.empty else None

        return {
            "Total Revenue": round(total_revenue, 2),
            "Total Profit": round(total_profit, 2),
            "Overall Profit Margin (%)": round(margin, 2),
            "Best Store": best_store["Store"] if best_store is not None else None,
            "Best Store Revenue": round(best_store["Revenue"], 2) if best_store is not None else None,
            "Worst Store": worst_store["Store"] if worst_store is not None else None,
            "Worst Store Revenue": round(worst_store["Revenue"], 2) if worst_store is not None else None,
            "Best Product": best_product["Product"] if best_product is not None else None,
            "Best Product Revenue": round(best_product["Revenue"], 2) if best_product is not None else None,
            "Worst Product": worst_product["Product"] if worst_product is not None else None,
            "Worst Product Revenue": round(worst_product["Revenue"], 2) if worst_product is not None else None,
            "Best Region": best_region["Region"] if best_region is not None else None,
            "Best Region Revenue": round(best_region["Revenue"], 2) if best_region is not None else None,
            "Worst Region": worst_region["Region"] if worst_region is not None else None,
            "Worst Region Revenue": round(worst_region["Revenue"], 2) if worst_region is not None else None,
            "Average Daily Revenue": round(avg_daily_revenue, 2),
        }

    # -------------------------
    # TIME SERIES
    # -------------------------

    def time_series_analysis(self):
        daily = self.df["Revenue"].resample("D").sum()
        rolling = daily.rolling(7).mean()
        monthly = daily.resample("ME").sum()
        return daily, rolling, monthly

    def trend_direction(self):
        daily = self.df["Revenue"].resample("D").sum()
        diff = daily.diff().mean()

        if diff > 0:
            return "Increasing"
        elif diff < 0:
            return "Decreasing"
        else:
            return "Stable"

    # -------------------------
    # VISUALIZATIONS
    # -------------------------

    def daily_rolling_plot(self):
        daily = self.df["Revenue"].resample("D").sum()
        rolling = daily.rolling(7).mean()

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(daily.index, daily, alpha=0.35, linewidth=1, label="Daily Revenue")
        ax.plot(rolling.index, rolling, linewidth=2.5, label="7-Day Rolling Avg")

        peak_date = daily.idxmax()
        peak_value = daily.max()

        ax.scatter(peak_date, peak_value, color="red", zorder=5)
        ax.annotate("Peak", xy=(peak_date, peak_value),
                    xytext=(0, 10), textcoords="offset points",
                    ha="center", fontsize=9, color="red")

        ax.set_title("Revenue Trend (Daily vs Rolling Average)", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue")
        ax.legend(frameon=False)

        ax.yaxis.set_major_formatter(FuncFormatter(currency_k))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def store_revenue_plot(self):
        store_rev = self.df.groupby("Store")["Revenue"].sum().sort_values()
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(store_rev.index, store_rev.values, color="#69b3a2", edgecolor="black")
        ax.set_title("Total Revenue by Store", fontweight="bold")
        ax.set_xlabel("Revenue")
        ax.set_ylabel("Store")
        ax.xaxis.set_major_formatter(FuncFormatter(currency_k))
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def product_revenue_plot(self):
        fig, ax = plt.subplots(figsize=(10, 5))

        avg_rev = self.df.groupby("Product")["Revenue"].mean().sort_values(ascending=False)
        top_products = avg_rev.head(2).index

        for product in self.df["Product"].unique():
            daily = self.df[self.df["Product"] == product]["Revenue"].resample("D").sum()
            if product in top_products:
                ax.plot(daily.index, daily, linewidth=2.5, label=product)
            else:
                ax.plot(daily.index, daily, alpha=0.25, linewidth=1)

        ax.set_title("Product Revenue Trends (Top Performers Highlighted)", fontweight="bold")
        ax.set_xlabel("Date")
        ax.set_ylabel("Revenue")
        ax.legend(frameon=False)

        ax.yaxis.set_major_formatter(FuncFormatter(currency_k))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        fig.autofmt_xdate()
        plt.tight_layout()
        plt.show()

    def profit_margin_histogram(self):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.hist(self.df["Profit_Margin"], bins=8, color="#ffb347", edgecolor='black')
        ax.set_title("Profit Margin Distribution", fontweight="bold")
        ax.set_xlabel("Profit Margin (%)")
        ax.set_ylabel("Frequency")
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    def monthly_revenue_plot(self):
        monthly = self.df["Revenue"].resample("ME").sum()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(monthly.index, monthly, linewidth=2.5, color="#00bfff")
        ax.set_title("Monthly Revenue Trend", fontweight="bold")
        ax.set_xlabel("Month")
        ax.set_ylabel("Revenue")
        ax.yaxis.set_major_formatter(FuncFormatter(currency_k))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
        ax.grid(True, linestyle=":", alpha=0.6)
        plt.tight_layout()
        plt.show()

    def weekend_vs_weekday_plot(self):
        weekend = self.df.groupby("Is_Weekend")["Revenue"].sum().reindex([0, 1], fill_value=0)
        labels = ["Weekday", "Weekend"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(labels, weekend.values, color="#ff7f0e", edgecolor="black")
        ax.set_title("Revenue: Weekday vs Weekend", fontweight="bold")
        ax.set_ylabel("Revenue")
        ax.yaxis.set_major_formatter(FuncFormatter(currency_k))
        ax.grid(axis='y', linestyle='--', alpha=0.6)
        plt.tight_layout()
        plt.show()

    # -------------------------
    # ADVANCED GROUPING INSIGHTS
    # -------------------------

    def advanced_grouping_insights(self):
        results = {}

        results["Store_Product_Performance"] = (
            self.df.groupby(["Store", "Product"]).agg(
                Revenue=("Revenue", "sum"),
                Units_Sold=("Units_Sold", "sum")
            ).sort_values(["Store", "Revenue"], ascending=[True, False])
        )

        results["Region_Performance"] = self.df.groupby("Region", as_index=False)["Revenue"].sum()

        weekend = self.df.groupby("Is_Weekend", as_index=False).agg(
            Total_Revenue=("Revenue", "sum"),
            Total_Profit=("Profit", "sum")
        )
        total_rev = weekend["Total_Revenue"].sum()
        weekend["Revenue_Percentage"] = weekend["Total_Revenue"] / total_rev * 100 if total_rev != 0 else 0
        weekend["Day_Type"] = weekend["Is_Weekend"].map({0: "Weekday", 1: "Weekend"})
        results["Weekend_vs_Weekday"] = weekend

        top_products = (
            self.df.groupby(["Store", "Product"])["Revenue"].sum()
            .groupby(level=0, group_keys=False)
            .nlargest(3)
            .reset_index()
        )
        results["Top_3_Products_per_Store"] = top_products

        lowest_margin = (
            self.df.groupby("Region")
            .apply(lambda x: x.loc[x["Profit_Margin"].idxmin(), ["Product", "Profit_Margin"]])
            .reset_index(level=0)
        )
        results["Lowest_Margin_Product_per_Region"] = lowest_margin

        results["Loss_Making_Products"] = self.df[self.df["Profit"] < 0][["Product", "Store", "Profit"]]

        return results
    
    
    def customer_analysis(self):
    
        customer = self.df.groupby("Customer").agg(
                total_spending=("Revenue", "sum"),
                total_orders=("Revenue", "count"),
                avg_order_value=("Revenue", "mean")
            )
    
         # CLV (simple version)
        customer["CLV"] = customer["total_spending"]
    
         # customer type
        customer["type"] = np.where(customer["total_orders"] > 1, "Repeat", "New")
    
        print("\nCustomer Table:\n", customer)
    
        self.customer_df = customer
        
        
    def rfm_analysis(self):
        
        self.df = self.df.reset_index()
        rfm = self.df.groupby("Customer").agg(
            recency=("Date", "max"),
            frequency=("Date", "count"),
            monetary=("Revenue", "sum")
        )
    
        # recency convert to days
        rfm["recency"] = (self.df["Date"].max() - rfm["recency"]).dt.days
    
        # bins
        n_bins = min(5, rfm["recency"].nunique())
    
        # scoring
        rfm["r_score"] = pd.qcut(
            rfm["recency"].rank(method="first"),
            q=n_bins,
            labels=list(range(n_bins, 0, -1))
        )
    
        rfm["f_score"] = pd.qcut(
            rfm["frequency"].rank(method="first"),
            q=n_bins,
            labels=list(range(1, n_bins+1))
        )
    
        rfm["m_score"] = pd.qcut(
            rfm["monetary"].rank(method="first"),
            q=n_bins,
            labels=list(range(1, n_bins+1))
        )
    
        # segmentation
        def segment(row):
            r = int(row["r_score"])
            f = int(row["f_score"])
        
            if (r == n_bins) and (f >= n_bins-1):
               return "Champions"
            elif (r >= n_bins-1):
               return "Loyal"
            elif (r <= 2):
               return "At Risk"
            else:
               return "Lost"
    
        rfm["segment"] = rfm.apply(segment, axis=1)
    
        print("\nRFM Table:\n", rfm)
     
        self.rfm = rfm
        
        
    
    def plot_rfm_segments(self):

        seg = self.rfm["segment"].value_counts()

        plt.figure(figsize=(6,4))
        plt.bar(seg.index, seg.values)

        plt.title("RFM Customer Segments")
        plt.xlabel("Segment")
        plt.ylabel("Number of Customers")
 
        plt.tight_layout()
        plt.show()
        
    
    def plot_clv_top_customers(self):

        top = self.customer_df.sort_values("CLV", ascending=False).head(5)

        plt.figure(figsize=(6,4))
        plt.bar(top.index.astype(str), top["CLV"])
        plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_k))

        plt.title("Top 5 Customers by CLV")
        plt.xlabel("Customer ID")
        plt.ylabel("CLV")

        plt.tight_layout()
        plt.show()
    
    
    def plot_rfm_scores(self):

        fig, ax = plt.subplots(1,3, figsize=(10,3))

        ax[0].hist(self.rfm["recency"])
        ax[0].set_title("Recency")

        ax[1].hist(self.rfm["frequency"])
        ax[1].set_title("Frequency")

        ax[2].hist(self.rfm["monetary"])
        ax[2].set_title("Monetary")
        ax[2].xaxis.set_major_formatter(FuncFormatter(currency_k))

        plt.tight_layout()
        plt.show()
       
       
    
if __name__ == "__main__":
    # -------------------------
    # Sample Data
    # -------------------------
    data = {
        "Date": pd.date_range("2024-01-01", periods=150, freq="D"),
        "Customer": np.random.randint(1, 30, 150),
        "Store": np.random.choice(["North", "South", "East", "West"], 150),
        "Region": np.random.choice(["Urban", "Rural"], 150),
        "Product": np.random.choice(["Laptop", "Mobile", "Tablet", "Accessories"], 150),
        "Units_Sold": np.random.randint(1, 40, 150),
        "Price": np.random.choice([100, 200, 400, 800], 150),
        "Cost": np.random.choice([60, 120, 300, 500], 150)
    }

    dashboard = RetailAnalyticsDashboard(data)

    # -------------------------
    # Basic & Professional EDA
    # -------------------------
    print("\n=== Basic EDA ===")
    dashboard.basic_eda()

    print("\n=== Professional EDA ===")
    prof_eda = dashboard.professional_eda()
    for k, v in prof_eda.items():
        print(f"\n{k}:\n{v}")

    # -------------------------
    # Clean Data
    # -------------------------
    dashboard.data_cleaning(drop_invalid_price=True)

    # -------------------------
    # KPI Dashboard
    # -------------------------
    print("\n=== KPI Dashboard ===")
    kpis = dashboard.generate_kpi_dashboard
    for k, v in kpis.items():
        print(f"{k}: {v}")

    # -------------------------
    # Time Series & Trend
    # -------------------------
    print("\n=== Time Series Trend ===")
    daily, rolling, monthly = dashboard.time_series_analysis()
    print("Trend Direction:", dashboard.trend_direction())

    # -------------------------
    # Visualizations
    # -------------------------
    print("\n=== Visualizations ===")
    dashboard.daily_rolling_plot()
    dashboard.store_revenue_plot()
    dashboard.product_revenue_plot()
    dashboard.profit_margin_histogram()
    dashboard.monthly_revenue_plot()
    dashboard.weekend_vs_weekday_plot()
    
    
    dashboard.customer_analysis()
    dashboard.rfm_analysis()
    
    
    dashboard.plot_rfm_segments()
    dashboard.plot_clv_top_customers()
    dashboard.plot_rfm_scores()


    # -------------------------
    # Advanced Grouping Insights
    # -------------------------
    print("\n=== Advanced Grouping Insights ===")
    adv_insights = dashboard.advanced_grouping_insights()
    for k, v in adv_insights.items():
        print(f"\n{k}:\n{v}")
















        
        