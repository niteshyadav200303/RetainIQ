# RetainIQ — Customer Churn & Lifetime Value Intelligence

<p align="center">
  <strong>Predict customer churn. Understand customer value. Make smarter retention decisions.</strong>
</p>

<p align="center">
  <a href="https://retainiq-jul8liic22pxmy8gemp8j8.streamlit.app/">
    <img src="https://img.shields.io/badge/🚀_LIVE_DEMO-Launch_RetainIQ-635BFF?style=for-the-badge" alt="Launch RetainIQ Live Demo">
  </a>
  <a href="https://github.com/niteshyadav200303/RetainIQ">
    <img src="https://img.shields.io/badge/GitHub-Source_Code-181717?style=for-the-badge&logo=github" alt="GitHub Repository">
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Machine_Learning-XGBoost-1F6FEB?style=flat-square" alt="Machine Learning">
  <img src="https://img.shields.io/badge/Explainable_AI-SHAP-8A2BE2?style=flat-square" alt="Explainable AI">
  <img src="https://img.shields.io/badge/Status-Deployed-16A34A?style=flat-square" alt="Deployed">
</p>

---

## Overview

**RetainIQ** is an end-to-end machine learning project that helps e-commerce businesses identify customers at risk of churning, estimate their future monetary value, and understand the factors influencing their predictions.

Unlike approaches that rely on historical purchasing totals alone, RetainIQ uses a strict temporal evaluation strategy to predict future customer behavior from information available before a defined cutoff date.

The project combines predictive modeling, survival analysis, explainable AI, and an interactive dashboard in one application.

**[Open the Live Application →](https://retainiq-jul8liic22pxmy8gemp8j8.streamlit.app/)**

## Key Features

| Feature | Description |
|---|---|
| Customer Churn Prediction | Estimates churn risk using Logistic Regression and XGBoost classification models. |
| Customer Lifetime Value | Predicts customer monetary value over a future 90-day observation window. |
| Survival Analysis | Uses Kaplan–Meier estimates and Cox Proportional Hazards modeling to analyze customer inactivity. |
| Explainable AI | Uses SHAP to help interpret individual predictions and overall model behavior. |
| Customer Segmentation | Analyzes RFM-based customer groups, including Champions, Loyal Customers, At Risk, and Hibernating. |
| Batch Prediction | Upload customer data through CSV, score multiple records, and export predictions. |
| Interactive Dashboard | Explore customer risk, model performance, prediction results, and analytical insights. |

## Dashboard Modules

RetainIQ organizes its analysis into four main modules:

- **Overview:** Executive metrics, customer segments, churn patterns, and survival analysis.
- **Single Prediction:** Interactive customer inputs, estimated churn probability, projected 90-day LTV, and SHAP-based explanations.
- **Batch Prediction:** CSV upload, bulk scoring, risk analysis, and downloadable results.
- **Model Insights:** Model comparisons, confusion matrices, ROC curves, and global feature explanations.

## Model Performance

The following are the evaluation results documented for this project.

### Churn Classification

| Metric | Logistic Regression | XGBoost |
|---|---:|---:|
| ROC-AUC | 0.7796 | 0.7854 |
| F1-Score | 0.7340 | 0.7559 |
| Recall | 0.6955 | 0.7513 |
| Precision | 0.7771 | 0.7604 |
| Accuracy | 0.7113 | 0.7220 |

XGBoost achieved a recall of **75.13%**, compared with 69.55% for Logistic Regression, on the documented test set. Recall is particularly relevant to retention because missed churners may represent lost opportunities for intervention.

### Customer Lifetime Value Prediction

| Metric | XGBoost Regressor |
|---|---:|
| Mean Absolute Error (MAE) | £483.41 |
| Root Mean Squared Error (RMSE) | £3,328.74 |
| R² Score | 0.2758 |

The LTV model uses a log-transformed target to handle the skewed distribution of customer spending. The error values should be considered alongside the substantial variation in retail purchase amounts.

### Survival Analysis

- **Kaplan–Meier Estimation:** Analyzes customer survival patterns across RFM segments.
- **Cox Proportional Hazards:** Models time to inactivity using customer behavior features.
- **Concordance Index:** 0.3226 in the documented evaluation.

These results are experimental benchmarks, not guarantees of future business performance.

## Dataset & Methodology

RetainIQ uses the **UCI Online Retail II** transaction dataset covering retail activity from 2009–2011.

The documented cleaning pipeline reports:

| Processing Stage | Records |
|---|---:|
| Raw transactions | 1,067,371 |
| Cleaned transactions | 779,425 |
| Distinct customers in cleaned data | 5,878 |

The preprocessing workflow removes records with missing customer identifiers, cancellation transactions, invalid quantities or prices, and duplicate entries.

### Leakage-Free Temporal Evaluation

The project uses a temporal cutoff of **September 1, 2011**, with a subsequent 90-day observation window ending November 30, 2011.

- **Feature window:** Customer behavior observed before the cutoff.
- **Churn target:** Whether the customer made zero qualifying purchases during the future observation window.
- **LTV target:** Customer monetary spending during that future window.

This approach is designed to avoid using future purchase information to construct historical predictive features.

Dataset source: [UCI Machine Learning Repository — Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii)

## Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core application and data processing |
| Pandas & NumPy | Data preparation and numerical analysis |
| Scikit-learn | Preprocessing, baseline modeling, and evaluation |
| XGBoost | Churn classification and LTV regression |
| Lifelines | Survival analysis |
| SHAP | Model explainability |
| Streamlit | Interactive web application |
| Plotly | Interactive data visualizations |
| Joblib | Model serialization and loading |

## Project Structure

```text
RetainIQ/
├── .streamlit/
│   └── config.toml
├── app/
│   ├── app.py
│   ├── components.py
│   └── style.css
├── data/
│   ├── raw/
│   └── processed/
├── models/
├── notebooks/
├── reports/
├── src/
├── .gitignore
├── README.md
└── requirements.txt
```

**Directory overview**

- `app/` — Streamlit entry point, reusable UI components, and styling.
- `data/` — Raw and processed datasets.
- `models/` — Serialized trained models and explainability artifacts.
- `notebooks/` — Exploratory analysis and experimentation.
- `reports/` — Evaluation results and project documentation.
- `src/` — Reusable data processing and modeling code.
- `.streamlit/` — Application configuration and theme settings.

## Run Locally

### Prerequisites

- Python 3.10 or later
- Git
- The project dependencies listed in `requirements.txt`

### Installation

**1. Clone the repository**

```bash
git clone https://github.com/niteshyadav200303/RetainIQ.git
cd RetainIQ
```

**2. Create a virtual environment**

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**4. Launch the application**

```bash
streamlit run app/app.py
```

Streamlit will display a local URL in your terminal. Open that URL in your browser.

> **Note:** Local execution requires the model artifacts and any other runtime files referenced by the application to be present at their expected paths. Large files or datasets excluded from Git must be obtained separately.

## Responsible Use & Limitations

- Predictions are estimates and should not be treated as certainties about individual customers.
- The model benchmarks reflect the dataset, feature definitions, temporal split, and evaluation procedure used in this project.
- Retail behavior can change over time, so production deployment would require monitoring, validation, and periodic retraining.
- SHAP explanations describe model behavior; they do not establish causal relationships.
- Customer data should be handled securely, with appropriate privacy safeguards.

## Future Improvements

- Automated model monitoring and drift detection.
- Configurable churn thresholds based on retention campaign costs.
- Customer-level retention recommendations.
- Time-based backtesting across multiple evaluation periods.
- A reproducible training pipeline and model versioning.
- Improved calibration of churn probabilities and survival estimates.

## About

RetainIQ is an individual machine learning project focused on customer analytics, predictive modeling, and explainable AI.

**Author:** Nitesh Yadav

**Repository:** [niteshyadav200303/RetainIQ](https://github.com/niteshyadav200303/RetainIQ)

**Live Demo:** [retainiq.streamlit.app](https://retainiq-jul8liic22pxmy8gemp8j8.streamlit.app/)

---

<p align="center">
  <strong>RetainIQ</strong><br>
  Turning customer behavior into actionable intelligence.
</p>
