# RetainIQ: Explainable E-Commerce Customer Churn Prediction and Forward Lifetime Value Estimation Using Leakage-Free Temporal Modeling

**A Final Year Individual Mini Project Report**  
**Course:** AI & DS II (Artificial Intelligence & Data Science II)  
**Author:** Nitesh  
**Academic Year:** 2025–2026  

---

## Abstract

In non-contractual e-commerce environments, customer attrition occurs without explicit cancellation signals. Conventional predictive frameworks frequently succumb to target leakage by deriving customer aggregations across non-stationary temporal windows, resulting in inflated performance estimates that collapse in operational deployment. This report presents **RetainIQ**, an end-to-end machine learning and survival analysis framework constructed on the benchmark UCI Online Retail II dataset (1.06 million raw transactions). By implementing a strictly leak-free temporal cutoff at September 1, 2011, customer features are computed solely from historical interactions, while ground-truth churn (binary departure over 90 days) and forward customer lifetime value (LTV gross spend) are observed exclusively in the subsequent 90-day window. We benchmark an imbalance-weighted XGBoost classifier against an L2-regularized Logistic Regression baseline, demonstrating superior discriminative performance for XGBoost with an **ROC-AUC of 0.7854**, **F1-score of 0.7559**, and a **recall of 75.13%** (+5.58% over baseline). Furthermore, survival dynamics are analyzed via Kaplan-Meier estimators and Cox Proportional Hazards, while forward LTV is projected via an XGBoost regressor (MAE = £483.41). Model transparency is established through local and global SHapley Additive exPlanations (SHAP), and the operational solution is deployed via an interactive, card-based SaaS dashboard.

---

## 1. Introduction

Customer acquisition costs in modern digital commerce have surged by over 60% across the preceding decade, establishing customer retention as a paramount determinant of platform enterprise value. In contractual settings (e.g., Netflix, SaaS subscriptions), churn is explicitly observable when a customer terminates an active billing agreement. Conversely, non-contractual retail platforms (e.g., Amazon, wholesale merchants) operate under silent attrition: customers simply cease placing orders without providing notice.

Predicting silent attrition presents significant machine learning challenges:
1. **Target and Feature Leakage:** Calculating recency, frequency, and monetary (RFM) metrics over full historical datasets conflates antecedent signals with future target windows.
2. **Class Imbalance and Decision Asymmetry:** Retaining an at-risk customer yields higher utility than marketing to an already loyal buyer; hence, recall is critical.
3. **Interpretability Deficits:** Black-box gradient boosting models must provide actionable behavioral rationales to empower commercial retention teams.

RetainIQ addresses these foundational challenges through an integrated data engineering, predictive modeling, survival analysis, and explainability architecture.

---

## 2. Literature Review

The academic and industrial literature on customer retention spans probabilistic models, supervised tree ensembles, and explainable AI:

1. **Fader, P. S., Hardie, B. G., & Lee, K. L. (2005).** *"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model.* Marketing Science, 24(2), 275-284.  
   *Contribution:* Introduced the BG/NBD (Beta-Geometric / Negative Binomial Distribution) framework, demonstrating that customer transaction frequency and dropout probability can be estimated parametrically.
2. **Chen, T., & Guestrin, C. (2016).** *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.  
   *Contribution:* Established scalable tree boosting with exact greedy search, sparsity-aware split finding, and custom loss regularization, forming the state-of-the-art for tabular enterprise data.
3. **Lundberg, S. M., & Lee, S. I. (2017).** *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 30), 4765-4774.  
   *Contribution:* Formulated SHAP (SHapley Additive exPlanations), unifying cooperative game theory with machine learning interpretability to deliver locally accurate and globally consistent feature attributions.
4. **Cox, D. R. (1972).** *Regression Models and Life-Tables.* Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187-202.  
   *Contribution:* Formulated the semi-parametric proportional hazards model, allowing time-to-event covariate analysis in the presence of right-censored observation windows.
5. **Kaplan, E. L., & Meier, P. (1958).** *Nonparametric Estimation from Incomplete Observations.* Journal of the American Statistical Association, 53(282), 457-481.  
   *Contribution:* Established the foundational non-parametric survival function estimator for longitudinal cohorts.
6. **Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012).** *New Insights into Churn Prediction in the Telecommunication Sector.* European Journal of Operational Research, 218(1), 211-229.  
   *Contribution:* Emphasized the critical necessity of profit-driven metric optimization and cost-sensitive classification over raw accuracy in customer churn forecasting.

---

## 3. Methodology & System Architecture

### 3.1 Data Acquisition and Quality Audit
The raw UCI Online Retail II dataset encompasses transactions occurring between December 1, 2009, and December 9, 2011 (1,067,371 rows). The cleaning protocol:
- **Guest Transaction Filtering:** Dropped 243,007 rows missing `CustomerID`.
- **Cancellation Filtering:** Dropped 18,744 invoices with code prefix `'C'`.
- **Anomalous Values:** Eliminated 71 entries with non-positive price or quantity.
- **De-duplication:** Discarded 26,124 duplicate entries, resulting in 779,425 verified transactions across 5,878 distinct customers.

### 3.2 Leakage-Free Temporal Cutoff Formulation
To mirror production deployment, a fixed cutoff date was designated:
$$\tau = \text{2011-09-01 00:00:00}$$

Features are extracted strictly from transactions $t < \tau$:
- **Recency ($R$):** Days elapsed between the customer's last purchase and $\tau$.
- **Frequency ($F$):** Total unique purchase invoices executed before $\tau$.
- **Monetary ($M$):** Cumulative historical monetary spend ($\sum \text{Qty} \times \text{Price}$) before $\tau$.
- **Tenure ($T$):** Days elapsed between the customer's initial transaction and $\tau$.
- **Average Order Value ($AOV$):** $M / F$.
- **Unique Products:** Cardinality of distinct stock codes purchased.
- **Geographic Indicator ($IsUK$):** Binary indicator ($1$ for United Kingdom, $0$ for international).

Labels are derived strictly from the forward observation window $[\tau, \tau + 90\text{ days})$:
$$Y_{\text{churn}} = \begin{cases} 1 & \text{if } \text{Invoices}_{[\tau, \tau + 90)} = 0 \\ 0 & \text{if } \text{Invoices}_{[\tau, \tau + 90)} \ge 1 \end{cases}$$
$$Y_{\text{ltv}} = \sum_{i \in [\tau, \tau+90)} \text{Spend}_i$$

Cohort size at cutoff comprises 5,249 customers, exhibiting a baseline churn rate of **57.31%**.

### 3.3 Modeling Formulations
1. **Baseline Classifier:** L2-penalized Logistic Regression with `StandardScaler` and `class_weight='balanced'`.
2. **Deployed Classifier:** XGBoost with `scale_pos_weight = N_{\text{retained}} / N_{\text{churned}} = 0.745` to prevent negative-class neglect.
3. **LTV Regressor:** XGBoost Regressor trained on logarithmic target $\log(1 + Y_{\text{ltv}})$ to stabilize heavy monetary right-skew.
4. **Survival Modeling:** Kaplan-Meier survival curves by RFM segments and Cox Proportional Hazards regression for time-to-purchase analysis.

---

## 4. Experimental Results & Discussion

### 4.1 Classification Benchmarks
Evaluated on a stratified holdout test partition ($N = 1,313$, seed 42):

| Metric | Logistic Regression (Baseline) | XGBoost Classifier (Deployed) | Relative Gain |
| :--- | :---: | :---: | :---: |
| **ROC-AUC** | 0.7796 | **0.7854** | +0.58% |
| **F1-Score** | 0.7340 | **0.7559** | +2.19% |
| **Recall (Sensitivity)** | 0.6955 | **0.7513** | **+5.58%** |
| **Precision** | **0.7771** | 0.7604 | -1.67% |
| **Accuracy** | 0.7113 | **0.7220** | +1.07% |

*Discussion:* In commercial churn management, false negatives (failing to identify an abandoning customer) are far more costly than false positives (offering a loyalty discount to a customer who would have purchased anyway). XGBoost captured 565 actual churners compared to 523 for Logistic Regression, representing an operational detection increase of 42 high-risk accounts in the test set alone.

### 4.2 Forward LTV Regression Performance
- **Mean Absolute Error (MAE):** $\pounds 483.41$
- **Root Mean Squared Error (RMSE):** $\pounds 3,328.74$
- **Coefficient of Determination ($R^2$):** $0.2758$  
*Discussion:* While raw retail monetary values exhibit extreme outlier dispersion (wholesalers with $> \pounds 20,000$ spend), logarithmic transformation stabilized convergence and generated actionable forward value estimates.

### 4.3 Survival Analysis Insights
- **Kaplan-Meier Trajectories:** Accounts categorized in the *Champions* and *Loyal Customers* segments maintained active survival probabilities above 0.75 through day 90. Conversely, *At Risk* and *Hibernating* cohorts experienced immediate steep drops within the first 21 days, validating the business recommendation for rapid re-engagement cadences.
- **Cox Proportional Hazards:** Recency exhibited a positive hazard coefficient ($\beta = +0.0026$), confirming that dormancy exponentially accelerates customer lapse risk.

### 4.4 Explainable AI (SHAP)
Global TreeExplainer attribution revealed:
1. **Recency:** Primary contributor to model output variance; recency values $> 60$ days caused massive positive SHAP shifts toward churn.
2. **Frequency & Unique Products:** Strong negative SHAP attribution, demonstrating that basket variety and frequent order cycles create strong retention moats.
3. **Average Order Value:** Secondary impact, confirming that engagement frequency outweighs transaction size when predicting customer longevity.

---

## 5. Visual Artifacts Summary

The pipeline generated 13 production figures archived in `reports/figures/`:
- `01_churn_distribution.png`: Cohort target distribution (57.3% churned vs 42.7% retained).
- `02_rfm_distributions.png`: Log-scaled RFM distributions demonstrating heavy tail characteristics.
- `03_correlation_heatmap.png`: Feature correlation matrix confirming recency's strong positive relationship with churn ($r = +0.38$).
- `04_churn_by_segment.png`: Segment-wise churn breakdown.
- `05_ltv_vs_monetary.png`: Historical vs forward monetary spend log-log scatter.
- `06_confusion_matrices.png`: Side-by-side confusion matrix comparisons.
- `07_roc_curves.png`: Multi-model ROC comparison curves.
- `08_shap_summary.png`: Global SHAP beeswarm importance plot.
- `09_kaplan_meier_survival.png`: Longitudinal survival curves across 5 RFM segments.
- `10_app_overview.png` to `13_app_model_insights.png`: Verified Streamlit SaaS UI views.

---

## 6. Conclusion & Future Work

The RetainIQ project successfully formulates, validates, and deploys an end-to-end customer churn and lifetime value prediction system. By enforcing a strict temporal split, the model prevents data leakage and delivers authentic commercial utility. XGBoost demonstrates strong predictive superiority, while SHAP interpretability and survival analysis provide clear operational insights.

**Future Work:**
1. **Sequence Modeling:** Applying Recurrent Neural Networks (LSTM) or Transformer architectures directly to raw transaction event sequences to capture seasonal buying rhythms.
2. **Dynamic Pricing Integration:** Integrating predicted forward LTV into an automated reinforcement learning engine to optimize personalized discount allocations.
3. **Production Streaming:** Wrapping feature engineering into a real-time event streaming pipeline (e.g., Apache Kafka / Feast feature store).

---

## 7. References

1. Fader, P. S., Hardie, B. G., & Lee, K. L. (2005). *"Counting Your Customers" the Easy Way: An Alternative to the Pareto/NBD Model.* Marketing Science, 24(2), 275-284.
2. Chen, T., & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794.
3. Lundberg, S. M., & Lee, S. I. (2017). *A Unified Approach to Interpreting Model Predictions.* Advances in Neural Information Processing Systems (NeurIPS 30), 4765-4774.
4. Cox, D. R. (1972). *Regression Models and Life-Tables.* Journal of the Royal Statistical Society: Series B (Methodological), 34(2), 187-202.
5. Kaplan, E. L., & Meier, P. (1958). *Nonparametric Estimation from Incomplete Observations.* Journal of the American Statistical Association, 53(282), 457-481.
6. Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). *New Insights into Churn Prediction in the Telecommunication Sector.* European Journal of Operational Research, 218(1), 211-229.
