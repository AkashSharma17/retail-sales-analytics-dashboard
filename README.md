# Retail Sales Analytics Dashboard

A production-style data analytics pipeline that processes retail sales data to surface actionable business insights — from revenue trends and category performance to customer segmentation and profitability analysis.

---

## Project Overview

This pipeline automates the full analytics lifecycle:

- Sales performance breakdown by product category
- Monthly revenue trend detection
- Customer value segmentation (Low / Medium / High)
- Profit margin analysis by product and category

---

## Key Features

- Automated end-to-end pipeline: Load → Clean → Analyze → Visualize
- Category-wise and monthly sales analysis
- Customer segmentation by lifetime value
- Profit margin calculation per product
- Professional chart generation (PNG)
- Structured analytics report output
- Comprehensive execution logging

---

## Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core runtime |
| Pandas | Data manipulation and analysis |
| NumPy | Numerical computation |
| Matplotlib | Chart generation |
| Seaborn | Statistical visualization |

---

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/AkashSharma17/retail-sales-analytics-dashboard.git
cd retail-sales-analytics-dashboard

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Pipeline

```bash
python main.py
```

---

## Output Structure

| Location | Contents |
|----------|---------|
| `output/visualizations/` | Generated charts (PNG) |
| `output/reports/` | Analytics summary report (TXT) |
| `output/data/` | Processed dataset (CSV) |
| `logs/` | Execution logs |

---

## Project Structure

```
retail-sales-analytics-dashboard/
├── src/
│   ├── main.py               # Pipeline orchestrator
│   ├── config.py             # Configuration and constants
│   ├── data_loader.py        # Data ingestion
│   ├── data_processor.py     # Cleaning and preprocessing
│   ├── analytics_engine.py   # Core analysis logic
│   ├── visualizer.py         # Chart generation
│   └── utils.py              # Helper utilities
├── data/
│   ├── raw/                  # Raw input data
│   └── processed/            # Cleaned output data
├── output/                   # Analysis results
├── logs/                     # Execution logs
├── main.py                   # Entry point
├── requirements.txt          # Python dependencies
├── .gitignore
└── README.md
```

---

## Input Data Format

The pipeline expects a CSV file with the following columns:

| Column | Format | Description |
|--------|--------|-------------|
| `Date` | `YYYY-MM-DD` | Transaction date |
| `Product` | String | Product name |
| `Category` | String | Product category |
| `Quantity` | Integer | Units sold |
| `Price` | Float | Unit selling price |
| `Cost` | Float | Unit cost |
| `Customer` | String | Customer name |

---

## Generated Outputs

### Charts
- **Sales by Category** — Bar chart of revenue per category
- **Monthly Revenue Trends** — Line chart tracking revenue over time
- **Top Customers** — Horizontal bar chart ranked by purchase value
- **Profit Distribution** — Histogram of profit margins

### Report Metrics
- Total Revenue and Total Profit
- Average Profit Margin
- Top-performing Products
- Category-level Performance Summary

---

## Customization

### Add a Custom Analysis

Edit `src/analytics_engine.py`:

```python
def analyze_custom_metric(self):
    """Define your custom analysis logic here."""
    pass
```

### Modify Visualizations

Edit `src/visualizer.py` to update chart styles, color schemes, or add new chart types.

### Update Configuration

Edit `src/config.py` to change file paths, segmentation thresholds, or pipeline settings.

---

## Skills & Concepts Covered

| Area | Details |
|------|---------|
| **Data Engineering** | End-to-end pipeline design from raw ingestion to structured output |
| **Data Cleaning** | Null handling, type normalization, and outlier treatment |
| **Feature Engineering** | Deriving business metrics from transactional fields |
| **Exploratory Analysis** | Statistical profiling, trend detection, and pattern identification |
| **Business Intelligence** | KPI computation across revenue, profit, and customer dimensions |
| **Data Visualization** | Chart design using Matplotlib and Seaborn |
| **Automated Reporting** | Generating structured, stakeholder-ready summary reports |
| **Project Architecture** | Modular, production-style Python project organization |

---

## Author

**Akash Sharma**  
GitHub: [@AkashSharma17](https://github.com/AkashSharma17)

---

## License

This project is open source and available under the [MIT License](LICENSE).