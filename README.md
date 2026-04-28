# 📊 Retail Analytics Dashboard

## 📌 Project Overview

The **Retail Analytics Dashboard** is a production-style data analytics project built using Python. It transforms raw retail transaction data into actionable business insights through a structured pipeline that includes data cleaning, feature engineering, exploratory data analysis (EDA), KPI computation, time-series analysis, and visualization.

The project is designed to simulate a real-world business intelligence system used in retail environments for decision-making and performance tracking.

---

## 🎯 Objective

The primary goal of this project is to analyze retail sales data and extract meaningful insights to support business decisions, including:

- Revenue and profitability tracking
- Customer behavior analysis
- Product and category performance evaluation
- Regional sales comparison
- Sales trend forecasting patterns

---

## 🧠 Key Features

### 📊 Data Processing Pipeline
- Automated CSV data ingestion
- Missing value detection and handling
- Duplicate record removal
- Data type validation and transformation

### 🧮 Feature Engineering
- Revenue calculation
- Profit estimation
- Time-based features (day, month, weekend flag)
- Discount impact analysis

### 📈 Business Intelligence (KPI Dashboard)
- Total Revenue
- Total Orders
- Total Customers
- Average Order Value (AOV)
- Best performing region
- Best product category

### ⏱ Time Series Analysis
- Daily revenue trends
- 7-day rolling average
- Monthly revenue aggregation
- Trend direction detection

### 👥 Customer Analytics
- Customer segmentation (Repeat vs New)
- Spending behavior analysis
- Customer-level performance metrics

### 📉 Data Visualization
- Revenue trend analysis
- Region-wise performance charts
- Product category comparison
- Time-series visual insights

### 🪵 Logging System
- Step-by-step execution tracking
- Error and warning logging
- Data quality monitoring

---

## 🛠 Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib
- Matplotlib Date Utilities
- Python Logging Module

---

## 📁 Dataset Schema

The project uses structured retail transaction data with the following fields:

- `order_id` – Unique order identifier  
- `order_date` – Transaction date  
- `product_id` – Product identifier  
- `product_category` – Category of product  
- `price` – Unit price  
- `quantity` – Number of units sold  
- `discount` – Discount applied  
- `customer_id` – Customer identifier  
- `region` – Sales region  
- `payment_method` – Payment type  

---

## 🔄 Project Workflow

1. Load dataset (CSV)
2. Data validation and cleaning
3. Feature engineering
4. Exploratory Data Analysis (EDA)
5. KPI dashboard generation
6. Time series analysis
7. Visualization generation
8. Customer behavior analysis
9. Export cleaned dataset

---

## 📊 Output Insights

This project generates the following business insights:

- Revenue trends over time
- High-performing regions and categories
- Customer purchasing behavior
- Sales seasonality patterns
- Profitability distribution

---

## 📦 Output Files

- `cleaned_retail_data.csv` → Processed dataset after cleaning and feature engineering

---

## 🚀 How to Run

```bash
git clone https://github.com/your-username/retail-analytics-dashboard.git

cd retail-analytics-dashboard

pip install pandas numpy matplotlib

python Retail_Analytics_Dashboard.py
