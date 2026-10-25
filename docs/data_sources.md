# Data Sources and Provenance

## 1. Overview and Business Context
This project utilizes a public customer churn dataset within the context of a fictional subscription case study: **"Northstar SaaS"**.
*Disclaimer: Northstar SaaS is an entirely fictional case study company designed for rigorous analytics and ML portfolio demonstration. No proprietary, internal, or private customer information from Northstar or any real corporation was collected or accessed.*

---

## 2. Dataset Attribution and Origin

| Attribute | Details |
| :--- | :--- |
| **Dataset Name** | IBM Telco Customer Churn |
| **Author / Publisher** | International Business Machines (IBM) Sample Data Resources |
| **Authoritative Repository** | [IBM Community / GitHub: telco-customer-churn-on-icp4d](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv) |
| **Mirror Archives** | Kaggle (BlastChar/telco-customer-churn), IBM Business Analytics Community |
| **Licence / Usage Terms** | Public Domain / Open Sample Dataset for Analytics & Education (Apache 2.0 / CC0 community mirrors) |
| **Access / Retrieval Date** | 2026-10-23 |
| **Local File Path** | `data/raw/Telco-Customer-Churn.csv` |
| **File Format** | Comma-Separated Values (CSV), UTF-8 encoded |
| **File Size** | 970,457 bytes (~948 KB) |
| **Row Count** | 7,043 rows |
| **Column Count** | 21 columns |
| **Target Column** | `Churn` (`"Yes"` / `"No"`) |

---

## 3. Dataset Characteristics and Structural Assessment

### Genuine Time Dimension Evaluation
- **Does the dataset possess a longitudinal or dynamic time dimension suitable for time-series forecasting?**
  **NO.** 
  The dataset does not contain event timestamp logs, observation cohort dates, monthly billing snapshots, or cancellation timestamp fields.
- **Nature of the Data:** 
  The dataset is a **cross-sectional retrospective snapshot**. Each row captures a customer record at a single static point in time, with aggregated tenure (months subscribed), contract specifications, subscribed features, monthly charges, and lifetime total charges.

### Known Limitations
1. **Retrospective Snapshot:** Because the data represents a single retrospective slice, dynamic survival trajectories over time must be approximated or framed carefully as a retrospective classification study rather than a prospective continuous forecasting engine.
2. **Coarse Aggregations:** Features like `TotalCharges` represent cumulative lifetime billing up to the moment of observation, introducing potential correlation with tenure (`tenure * MonthlyCharges ≈ TotalCharges`).
3. **Absence of Usage Telemetry:** While service types (streaming, security, tech support) are present, detailed behavioural logs (e.g. daily logins, feature engagement, support ticket volume) are not recorded.

---

## 4. Reproducible Download Procedure

To programmatically fetch the dataset into the repository:
```bash
python -c "from src.data_loader import download_raw_dataset; download_raw_dataset()"
```

Or manually:
1. Navigate to: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`
2. Save the file directly as `data/raw/Telco-Customer-Churn.csv`.
3. Verify that the file size is ~970 KB and has 7,043 rows.
