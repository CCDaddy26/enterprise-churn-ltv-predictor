# Data Dictionary: Northstar SaaS Churn & LTV Predictor

This dictionary defines each feature in the raw dataset (`data/raw/Telco-Customer-Churn.csv`), its empirical data type, semantics, cardinality, missing-data handling policy, and role in model evaluation.

---

## 1. Primary Identifier & Target

| Column Name | Raw Type | Clean Type | Description | Values / Range | Preprocessing / Modeling Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `customerID` | `object` | `string` | Unique customer alphanumeric identifier | 7,043 unique alphanumeric strings (e.g., `7590-VHVEG`) | **Excluded from predictors.** Retained for record tracing only. |
| `Churn` | `object` | `category` / `int` | Primary target indicating customer cancellation | `"No"` (5,174; 73.46%), `"Yes"` (1,869; 26.54%) | **Supervised Target ($y$).** Mapped to binary integer (`1` for Churn, `0` for Retained). |

---

## 2. Demographic & Account Features

| Column Name | Raw Type | Clean Type | Description | Values / Categories | Preprocessing / Modeling Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `gender` | `object` | `category` | Customer biological gender | `"Female"` (3,488), `"Male"` (3,555) | One-hot encoded (`drop="first"`). |
| `SeniorCitizen` | `int64` | `category` | Indicator whether customer is $\ge 65$ years old | `0` (5,901; 83.78%), `1` (1,142; 16.22%) | Categorical factor; one-hot encoded. |
| `Partner` | `object` | `category` | Whether customer has a spouse / partner | `"Yes"` (3,402), `"No"` (3,641) | One-hot encoded (`drop="first"`). |
| `Dependents` | `object` | `category` | Whether customer has dependents | `"No"` (4,933), `"Yes"` (2,110) | One-hot encoded (`drop="first"`). |

---

## 3. Contract & Account Tenure Features

| Column Name | Raw Type | Clean Type | Description | Values / Range | Preprocessing / Modeling Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `tenure` | `int64` | `int` | Number of months the customer has stayed with company | Range: 0 to 72 months (Median: 29.0, Mean: 32.37) | Continuous feature; median-imputed, standard scaled. |
| `Contract` | `object` | `category` | The contract commitment term | `"Month-to-month"` (3,875), `"One year"` (1,473), `"Two year"` (1,695) | One-hot encoded. Critical business segmentation factor. |
| `PaperlessBilling` | `object` | `category` | Whether electronic billing is enabled | `"Yes"` (4,171), `"No"` (2,872) | One-hot encoded (`drop="first"`). |
| `PaymentMethod` | `object` | `category` | Customer's primary payment method | `"Electronic check"` (2,365), `"Mailed check"` (1,612), `"Bank transfer (automatic)"` (1,544), `"Credit card (automatic)"` (1,522) | One-hot encoded. Key operational segment. |

---

## 4. Product & Service Add-On Features

| Column Name | Raw Type | Clean Type | Description | Values / Categories | Preprocessing / Modeling Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `PhoneService` | `object` | `category` | Whether phone service is subscribed | `"Yes"` (6,361), `"No"` (682) | One-hot encoded. |
| `MultipleLines` | `object` | `category` | Multiple telephone lines status | `"No"` (3,390), `"Yes"` (2,971), `"No phone service"` (682) | One-hot encoded. |
| `InternetService` | `object` | `category` | Internet connection architecture | `"Fiber optic"` (3,096), `"DSL"` (2,421), `"No"` (1,826) | One-hot encoded. Major churn variance driver. |
| `OnlineSecurity` | `object` | `category` | Add-on security module status | `"No"` (3,498), `"Yes"` (2,019), `"No internet service"` (1,826) | One-hot encoded. |
| `OnlineBackup` | `object` | `category` | Add-on cloud backup module | `"No"` (3,088), `"Yes"` (2,429), `"No internet service"` (1,826) | One-hot encoded. |
| `DeviceProtection`| `object` | `category` | Add-on warranty / device protection | `"No"` (3,095), `"Yes"` (2,422), `"No internet service"` (1,826) | One-hot encoded. |
| `TechSupport` | `object` | `category` | Dedicated tech support assistance | `"No"` (3,473), `"Yes"` (2,044), `"No internet service"` (1,826) | One-hot encoded. |
| `StreamingTV` | `object` | `category` | Television streaming bundle | `"No"` (2,810), `"Yes"` (2,707), `"No internet service"` (1,826) | One-hot encoded. |
| `StreamingMovies`| `object` | `category` | Movie streaming bundle | `"No"` (2,785), `"Yes"` (2,732), `"No internet service"` (1,826) | One-hot encoded. |

---

## 5. Billing & Revenue Features

| Column Name | Raw Type | Clean Type | Description | Values / Range | Preprocessing / Modeling Role |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `MonthlyCharges` | `float64` | `float64` | Current amount billed to customer monthly | Range: $18.25 to $118.75 (Median: $70.35, Mean: $64.76) | Continuous feature; median-imputed, standard scaled. |
| `TotalCharges` | `object` | `float64` | Cumulative charges billed over customer's lifetime | Range: $18.80 to $8,684.80 (Median: $1,397.47, Mean: $2,279.73). **Contains 11 blank strings (`" "`).** | Cleaned by stripping whitespace; blank strings correspond to `tenure=0` and are set to `0.0`. Median-imputed, standard scaled. |

---

## 6. Audit & Data Cleaning Policy Summary

1. **Zero Row Deletions:** All 7,043 rows are retained. No customer record is silently discarded.
2. **Whitespace Anomaly in `TotalCharges`:** Exactly 11 customers have a whitespace string in `TotalCharges`. All 11 records have `tenure = 0` (brand-new accounts billed 0 months). Imputing `0.0` for these accounts reflects physical domain reality without corrupting distributions.
3. **Identifier Isolation:** `customerID` is confirmed unique across all 7,043 rows (0 duplicates) and is strictly removed prior to feature matrix transformation to prevent overfitting.
