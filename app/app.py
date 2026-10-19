"""
RetainIQ - Customer Churn & LTV Predictor
An executive-grade SaaS web application for e-commerce customer retention intelligence.

Core Features:
1. Customer Risk Predictor: Interactive profile simulator with 3 pre-configured demo scenarios (Low, Medium, High Risk),
   instant Churn & 90-day LTV predictions, retention action recommendations, and plain-English explainability.
2. Batch CSV Scoring: Upload or 1-click test with 3 pre-built demo cohorts (VIP, At-Risk, Mixed),
   automatic risk scoring, KPI summaries, and CSV download.
3. Executive Overview: Cohort KPIs, segment churn rates, risk distribution, and Kaplan-Meier survival curves.
4. Model Insights: Rigorous benchmarks (XGBoost vs Logistic Regression), confusion matrix, ROC curves, and SHAP.
"""

import inspect
import io
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

try:
    from app.components import (
        card,
        inject_custom_css,
        kpi_card,
        recommendation_card,
        render_footer,
        render_header,
        risk_badge,
    )
except ModuleNotFoundError:
    from components import (
        card,
        inject_custom_css,
        kpi_card,
        recommendation_card,
        render_footer,
        render_header,
        risk_badge,
    )

# Page configuration
st.set_page_config(
    page_title="RetainIQ - Customer Churn & LTV Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Dark / Black Mode State
if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = False

MODELS_DIR = ROOT_DIR / "models"
PROCESSED_DIR = ROOT_DIR / "data/processed"
DEMO_DIR = ROOT_DIR / "data/demo"


# -------------------------------------------------------------
# CACHED RESOURCE LOADERS
# -------------------------------------------------------------
@st.cache_resource
def load_all_models():
    """Load and cache ML artifacts."""
    try:
        lr_pipeline = joblib.load(MODELS_DIR / "lr_churn_pipeline.joblib")
        xgb_churn = joblib.load(MODELS_DIR / "xgb_churn_model.joblib")
        xgb_ltv = joblib.load(MODELS_DIR / "xgb_ltv_model.joblib")
        cph = joblib.load(MODELS_DIR / "cox_ph_model.joblib")
        kmf_dict = joblib.load(MODELS_DIR / "kmf_segments.joblib")
        explainer = joblib.load(MODELS_DIR / "shap_explainer.joblib")
        return {
            "lr": lr_pipeline,
            "xgb_churn": xgb_churn,
            "xgb_ltv": xgb_ltv,
            "cph": cph,
            "kmf_dict": kmf_dict,
            "explainer": explainer,
        }
    except Exception as e:
        st.error(f"Error loading models: {e}. Please ensure model training has completed.")
        return None


@st.cache_data
def load_customer_data():
    """Load processed customer feature dataset and metrics."""
    data_path = PROCESSED_DIR / "customer_features.parquet"
    if not data_path.exists():
        data_path = PROCESSED_DIR / "customer_features.csv"
    if data_path.name.endswith(".parquet"):
        df = pd.read_parquet(data_path)
    else:
        df = pd.read_csv(data_path)

    metrics_path = MODELS_DIR / "metrics.json"
    metrics = {}
    if metrics_path.exists():
        with open(metrics_path, "r") as f:
            metrics = json.load(f)
    return df, metrics


models = load_all_models()
df_customers, metrics_data = load_customer_data()


# -------------------------------------------------------------
# HELPER FUNCTIONS
# -------------------------------------------------------------
def get_risk_category(prob: float) -> str:
    if prob < 0.35:
        return "Low"
    elif prob < 0.65:
        return "Medium"
    else:
        return "High"


def predict_single(features_dict: dict):
    """Run model inference for a single customer profile."""
    cols = ["Recency", "Frequency", "Monetary", "Tenure", "AvgOrderValue", "UniqueProducts", "IsUK"]
    row_df = pd.DataFrame([features_dict])[cols]

    # Churn probability
    prob = float(models["xgb_churn"].predict_proba(row_df)[0, 1])
    risk = get_risk_category(prob)

    # LTV Prediction (invert log1p)
    ltv_log = float(models["xgb_ltv"].predict(row_df)[0])
    ltv = float(np.expm1(max(0, ltv_log)))

    # SHAP local values
    shap_vals = models["explainer"](row_df)

    return prob, risk, ltv, shap_vals, row_df


# -------------------------------------------------------------
# DEMO DATA PRESETS CONFIGURATION (NO EMOJIS)
# -------------------------------------------------------------
DEMO_PRESETS = {
    1: {
        "title": "01 | Loyal Champion",
        "badge": "Low Churn Risk",
        "description": "High-frequency buyer who recently made a large purchase. Strong brand loyalty and highest projected LTV.",
        "recency": 12,
        "frequency": 18,
        "monetary": 4500.0,
        "tenure": 365,
        "avg_order_value": 250.0,
        "unique_products": 85,
        "country_choice": "United Kingdom",
    },
    2: {
        "title": "02 | At-Risk Regular",
        "badge": "Medium Churn Risk",
        "description": "Previously consistent buyer whose last order was 4 months ago. Needs a proactive re-engagement push.",
        "recency": 120,
        "frequency": 4,
        "monetary": 650.0,
        "tenure": 240,
        "avg_order_value": 162.5,
        "unique_products": 25,
        "country_choice": "United Kingdom",
    },
    3: {
        "title": "03 | Lost / Hibernating",
        "badge": "High Churn Risk",
        "description": "One-time purchaser inactive for 8 months with low historical spend. High probability of defection.",
        "recency": 240,
        "frequency": 1,
        "monetary": 65.0,
        "tenure": 240,
        "avg_order_value": 65.0,
        "unique_products": 3,
        "country_choice": "International / Other",
    },
}


def apply_demo_preset(preset_id: int):
    """Callback to populate session state with demo values."""
    p = DEMO_PRESETS[preset_id]
    st.session_state["active_demo_id"] = preset_id
    st.session_state["active_demo_name"] = p["title"]

    st.session_state["recency"] = p["recency"]
    st.session_state["slider_recency"] = p["recency"]

    st.session_state["frequency"] = p["frequency"]
    st.session_state["input_frequency"] = p["frequency"]

    st.session_state["monetary"] = p["monetary"]
    st.session_state["input_monetary"] = p["monetary"]

    st.session_state["tenure"] = p["tenure"]
    st.session_state["slider_tenure"] = p["tenure"]

    st.session_state["avg_order_value"] = p["avg_order_value"]
    st.session_state["input_aov"] = p["avg_order_value"]

    st.session_state["unique_products"] = p["unique_products"]
    st.session_state["input_unique"] = p["unique_products"]

    st.session_state["country_choice"] = p["country_choice"]
    st.session_state["select_country"] = p["country_choice"]


# Initialize session state with Demo 1 if not present
if "active_demo_id" not in st.session_state:
    apply_demo_preset(1)


# -------------------------------------------------------------
# SIDEBAR: PRODUCT DETAILS & BLACK/DARK MODE TOGGLE
# -------------------------------------------------------------
with st.sidebar:
    st.markdown(
        inspect.cleandoc("""
        <div style="padding: 10px 0 16px 0;">
            <div style="font-size: 1.5rem; font-weight: 700; color: #0F172A; letter-spacing: -0.025em;">
                RetainIQ
            </div>
            <div style="font-size: 0.84rem; color: #64748B; font-weight: 500; margin-top: 4px;">
                Customer Churn & LTV Intelligence
            </div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    # Black / Dark Mode Switch (no emoji)
    dark_mode_active = st.toggle("Dark Mode", value=st.session_state["dark_mode"], key="dark_toggle")
    if dark_mode_active != st.session_state["dark_mode"]:
        st.session_state["dark_mode"] = dark_mode_active
        st.rerun()

    st.markdown("---")
    st.markdown(
        inspect.cleandoc(f"""
        <div style="background: {'#111827' if dark_mode_active else '#F8FAFC'}; border: 1px solid {'#1F2937' if dark_mode_active else '#E2E8F0'}; border-radius: 10px; padding: 14px; margin-bottom: 16px;">
            <div style="font-size: 0.75rem; font-weight: 700; color: #64748B; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 8px;">SYSTEM STATUS</div>
            <div style="font-size: 0.88rem; color: {'#F8FAFC' if dark_mode_active else '#0F172A'}; margin-bottom: 4px;">• <strong>Pipeline:</strong> XGBoost + SHAP (Active)</div>
            <div style="font-size: 0.88rem; color: {'#F8FAFC' if dark_mode_active else '#0F172A'}; margin-bottom: 4px;">• <strong>Cohort Tracked:</strong> {len(df_customers):,} Accounts</div>
            <div style="font-size: 0.88rem; color: {'#F8FAFC' if dark_mode_active else '#0F172A'};">• <strong>Forecast Horizon:</strong> 90 Days</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    st.markdown(
        inspect.cleandoc(f"""
        <div style="background: {'#111827' if dark_mode_active else '#F1F5F9'}; border: 1px solid {'#1F2937' if dark_mode_active else 'transparent'}; border-radius: 8px; padding: 12px 14px; font-size: 0.82rem; color: {'#94A3B8' if dark_mode_active else '#334155'}; line-height: 1.5;">
            <strong>Ready-to-Use Demo Data:</strong><br>
            • <strong>3 Customer Scenarios:</strong> Click Preset 01, 02, or 03.<br>
            • <strong>3 Batch Datasets:</strong> 1-click batch cohort scoring.<br>
            • <strong>Instant Results:</strong> Zero setup required.
        </div>
        """),
        unsafe_allow_html=True,
    )

# Inject custom CSS (with dark mode class applied if enabled)
inject_custom_css(dark_mode=dark_mode_active)

if dark_mode_active:
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #080B11 !important;
            color: #F1F5F9 !important;
        }
        .main .block-container {
            background-color: #080B11 !important;
        }
        p, span, label, div {
            color: #E2E8F0;
        }
        h1, h2, h3, h4, .app-title {
            color: #FFFFFF !important;
        }
        .card-header-title {
            color: #FFFFFF !important;
        }
        .kpi-value {
            color: #FFFFFF !important;
        }
        section[data-testid="stSidebar"] {
            background-color: #0B0F19 !important;
            border-right: 1px solid #1F2937 !important;
        }
        .retain-card {
            background-color: #111827 !important;
            border: 1px solid #1F2937 !important;
        }
        .kpi-container {
            background-color: #111827 !important;
            border: 1px solid #1F2937 !important;
        }
        .demo-preset-banner {
            background: linear-gradient(135deg, #111827 0%, #0F172A 100%) !important;
            border: 1px solid #1F2937 !important;
        }
        div.stButton > button[kind="secondary"] {
            background-color: #111827 !important;
            color: #F8FAFC !important;
            border: 1px solid #334155 !important;
        }
        div.stButton > button[kind="secondary"]:hover {
            border-color: #818CF8 !important;
            color: #818CF8 !important;
            background-color: #1E293B !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# Chart styling variables
text_color = "#FFFFFF" if dark_mode_active else "#0F172A"
grid_color = "#1E293B" if dark_mode_active else "#E2E8F0"
paper_bg = "rgba(0,0,0,0)"


# -------------------------------------------------------------
# TOP BRAND BANNER & MAIN NAVIGATION TABS (NO EMOJIS)
# -------------------------------------------------------------
st.markdown(
    inspect.cleandoc(f"""
    <div style="margin-bottom: 18px;">
        <div style="font-size: 1.85rem; font-weight: 800; color: {'#FFFFFF' if dark_mode_active else '#0F172A'}; letter-spacing: -0.03em;">
            RetainIQ Dashboard
        </div>
        <div style="font-size: 0.96rem; color: {'#94A3B8' if dark_mode_active else '#64748B'}; margin-top: 4px;">
            Enterprise e-commerce customer retention & 90-day forward lifetime value platform.
        </div>
    </div>
    """),
    unsafe_allow_html=True,
)

tab_pred, tab_batch, tab_overview, tab_insights = st.tabs([
    "Customer Risk Predictor",
    "Batch CSV Scoring",
    "Executive Overview",
    "Model Benchmarks & Explainability",
])


# =============================================================
# TAB 1: INTERACTIVE CUSTOMER RISK PREDICTOR
# =============================================================
with tab_pred:
    # 3 Pre-Made Demo Data Quick-Select Box with Digits (No Emojis)
    st.markdown(
        inspect.cleandoc(f"""
        <div class="demo-preset-banner">
            <div class="demo-banner-title" style="color: {'#FFFFFF' if dark_mode_active else '#0F172A'};">Pre-Configured Customer Profiles</div>
            <div class="demo-banner-desc" style="color: {'#94A3B8' if dark_mode_active else '#64748B'};">Select any scenario below to instantly populate metrics and observe how the predictive engine calculates risk:</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button("01 | Loyal Champion (Low Risk)", use_container_width=True, help="Active high-spend customer with low churn risk."):
            apply_demo_preset(1)
            st.rerun()
    with d2:
        if st.button("02 | At-Risk Regular (Medium Risk)", use_container_width=True, help="Decreasing frequency customer showing early signs of churn."):
            apply_demo_preset(2)
            st.rerun()
    with d3:
        if st.button("03 | Lost Account (High Risk)", use_container_width=True, help="Dormant buyer inactive for 8 months with high churn risk."):
            apply_demo_preset(3)
            st.rerun()

    # Notification of currently active preset
    active_name = st.session_state.get("active_demo_name", "01 | Loyal Champion")
    active_desc = DEMO_PRESETS.get(st.session_state.get("active_demo_id", 1), {}).get("description", "")
    st.markdown(
        inspect.cleandoc(f"""
        <div style="background: {'#111827' if dark_mode_active else '#FFFFFF'}; border: 1px solid {'#1F2937' if dark_mode_active else '#E2E8F0'}; border-left: 4px solid #635BFF; border-radius: 8px; padding: 10px 16px; margin-bottom: 20px; font-size: 0.86rem;">
            <strong style="color:{'#FFFFFF' if dark_mode_active else '#0F172A'};">Active Profile:</strong> <span style="color:#818CF8; font-weight:700;">{active_name}</span> &bull; <em style="color:{'#94A3B8' if dark_mode_active else '#64748B'};">{active_desc}</em>
        </div>
        """),
        unsafe_allow_html=True,
    )

    col_input, col_output = st.columns([1, 1], gap="large")

    with col_input:
        with card("Customer Attributes", "Review the customer's metrics or adjust any value below:"):
            c_a, c_b = st.columns(2)

            with c_a:
                recency = st.slider(
                    "Recency (Days since last order)",
                    min_value=1,
                    max_value=365,
                    value=int(st.session_state.get("recency", 12)),
                    key="slider_recency",
                    help="Fewer days indicates recent engagement.",
                )
                frequency = st.number_input(
                    "Order Frequency (Total orders)",
                    min_value=1,
                    max_value=150,
                    value=int(st.session_state.get("frequency", 18)),
                    key="input_frequency",
                    help="Total distinct purchase invoices before the cutoff date.",
                )
                monetary = st.number_input(
                    "Total Monetary Spend (£)",
                    min_value=5.0,
                    max_value=50000.0,
                    value=float(st.session_state.get("monetary", 4500.0)),
                    step=50.0,
                    key="input_monetary",
                    help="Cumulative historical gross spend.",
                )
                tenure = st.slider(
                    "Customer Tenure (Days with brand)",
                    min_value=1,
                    max_value=730,
                    value=int(st.session_state.get("tenure", 365)),
                    key="slider_tenure",
                    help="Days elapsed since the customer's very first purchase.",
                )

            with c_b:
                avg_order_value = st.number_input(
                    "Avg Order Value (£)",
                    min_value=1.0,
                    max_value=10000.0,
                    value=float(st.session_state.get("avg_order_value", 250.0)),
                    step=10.0,
                    key="input_aov",
                    help="Average spend per purchase transaction.",
                )
                unique_products = st.number_input(
                    "Unique Products Bought",
                    min_value=1,
                    max_value=1000,
                    value=int(st.session_state.get("unique_products", 85)),
                    step=1,
                    key="input_unique",
                    help="Product catalogue breadth explored by this customer.",
                )
                country_opts = ["United Kingdom", "International / Other"]
                curr_country = st.session_state.get("country_choice", "United Kingdom")
                country_idx = 0 if curr_country == "United Kingdom" else 1
                country_choice = st.selectbox(
                    "Customer Country",
                    country_opts,
                    index=country_idx,
                    key="select_country",
                    help="UK accounts account for 91% of baseline store activity.",
                )
                is_uk = 1 if country_choice == "United Kingdom" else 0

            st.caption("Adjust sliders or inputs above to evaluate custom customer situations in real-time.")

    # Execute Prediction
    input_features = {
        "Recency": recency,
        "Frequency": frequency,
        "Monetary": monetary,
        "Tenure": tenure,
        "AvgOrderValue": avg_order_value,
        "UniqueProducts": unique_products,
        "IsUK": is_uk,
    }

    prob, risk, ltv, shap_res, row_df = predict_single(input_features)

    with col_output:
        with card("RetainIQ AI Assessment", "Real-time risk scoring and forward value projection:"):
            # Top Score Row
            m1, m2 = st.columns(2)
            with m1:
                st.markdown(
                    inspect.cleandoc(f"""
                    <div style="font-size:0.8rem; font-weight:700; color:{'#94A3B8' if dark_mode_active else '#64748B'}; text-transform:uppercase; letter-spacing:0.04em;">RISK CLASSIFICATION</div>
                    <div style="margin: 6px 0 10px 0;">{risk_badge(risk)}</div>
                    <div style="font-size:2.1rem; font-weight:800; color:{'#FFFFFF' if dark_mode_active else '#0F172A'}; line-height:1.1;">{prob*100:.1f}%</div>
                    <div style="font-size:0.82rem; color:{'#94A3B8' if dark_mode_active else '#64748B'};">Churn Probability (Next 90 Days)</div>
                    """),
                    unsafe_allow_html=True,
                )
            with m2:
                st.markdown(
                    inspect.cleandoc(f"""
                    <div style="font-size:0.8rem; font-weight:700; color:{'#94A3B8' if dark_mode_active else '#64748B'}; text-transform:uppercase; letter-spacing:0.04em;">PROJECTED 90-DAY LTV</div>
                    <div style="font-size:2.1rem; font-weight:800; color:#818CF8; margin-top:28px; line-height:1.1;">£{ltv:.2f}</div>
                    <div style="font-size:0.82rem; color:{'#94A3B8' if dark_mode_active else '#64748B'};">Expected Forward Spend</div>
                    """),
                    unsafe_allow_html=True,
                )

            # Churn Probability Visual Gauge
            gauge_bar_color = "#059669" if prob < 0.35 else ("#D97706" if prob < 0.65 else "#DC2626")
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"family": "Inter", "size": 24, "color": text_color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#64748B"},
                        "bar": {"color": gauge_bar_color},
                        "bgcolor": "#1E293B" if dark_mode_active else "white",
                        "borderwidth": 1,
                        "bordercolor": grid_color,
                        "steps": [
                            {"range": [0, 35], "color": "#064E3B" if dark_mode_active else "#ECFDF5"},
                            {"range": [35, 65], "color": "#78350F" if dark_mode_active else "#FFFBEB"},
                            {"range": [65, 100], "color": "#7F1D1D" if dark_mode_active else "#FEF2F2"},
                        ],
                    },
                )
            )
            fig_gauge.update_layout(
                height=160,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor=paper_bg,
                font=dict(family="Inter", color=text_color),
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Actionable Retention Recommendation Box
            recommendation_card(risk, ltv)

    # Explainability Section (Plain English + SHAP, No Emojis)
    with card("Why did the AI make this prediction? (Explainable AI)", "Deconstructing the key behavioral signals driving this outcome:"):
        col_exp_text, col_exp_chart = st.columns([1.1, 1], gap="medium")

        values_arr = shap_res.values[0]
        feature_names = ["Recency", "Frequency", "Monetary", "Tenure", "AvgOrderValue", "UniqueProducts", "IsUK"]
        shap_df = pd.DataFrame({"Feature": feature_names, "SHAP_Impact": values_arr}).sort_values(by="SHAP_Impact", key=abs, ascending=False)

        with col_exp_text:
            st.markdown(f"<div style='font-weight:700; font-size:0.95rem; color:{'#FFFFFF' if dark_mode_active else '#0F172A'}; margin-bottom:10px;'>Top Decision Drivers:</div>", unsafe_allow_html=True)
            for idx, (_, row) in enumerate(shap_df.head(4).iterrows(), start=1):
                feat = row["Feature"]
                val = row["SHAP_Impact"]
                is_increasing = val > 0
                direction = "Increases Risk" if is_increasing else "Lowers Risk"
                status_color = "#F87171" if is_increasing else "#34D399"

                if feat == "Recency":
                    explanation = f"Last order was {recency} days ago. {'Dormancy strongly elevates churn risk.' if recency > 60 else 'Recent activity maintains high retention.'}"
                elif feat == "Frequency":
                    explanation = f"Customer has placed {frequency} orders. {'High order history cements loyalty.' if frequency >= 5 else 'Low order count leaves customer vulnerable.'}"
                elif feat == "Monetary":
                    explanation = f"Total spend of £{monetary:,.2f}. {'Substantial historical value.' if monetary > 1000 else 'Modest account spend.'}"
                elif feat == "Tenure":
                    explanation = f"Account age is {tenure} days with the brand."
                else:
                    explanation = f"Metric attribution effect: {abs(val):.3f} log-odds."

                card_bg = "#1E293B" if dark_mode_active else "#F8FAFC"
                card_border = "#334155" if dark_mode_active else "#E2E8F0"
                title_col = "#FFFFFF" if dark_mode_active else "#0F172A"
                sub_col = "#94A3B8" if dark_mode_active else "#64748B"

                st.markdown(
                    inspect.cleandoc(f"""
                    <div style="background:{card_bg}; border:1px solid {card_border}; border-radius:8px; padding:10px 14px; margin-bottom:8px; font-size:0.87rem;">
                        <span style="font-weight:700; color:{title_col};">{feat}:</span> 
                        <span style="color:{status_color}; font-weight:700;">{direction}</span><br>
                        <span style="color:{sub_col}; font-size:0.82rem;">{explanation}</span>
                    </div>
                    """),
                    unsafe_allow_html=True,
                )

        with col_exp_chart:
            st.markdown(f"<div style='font-weight:700; font-size:0.95rem; color:{'#FFFFFF' if dark_mode_active else '#0F172A'}; margin-bottom:10px;'>SHAP Feature Attribution Bar:</div>", unsafe_allow_html=True)
            shap_chart_df = shap_df.sort_values(by="SHAP_Impact", ascending=True)
            bar_colors = ["#DC2626" if v > 0 else "#059669" for v in shap_chart_df["SHAP_Impact"]]
            fig_shap = px.bar(
                shap_chart_df,
                x="SHAP_Impact",
                y="Feature",
                orientation="h",
                labels={"SHAP_Impact": "Impact (Red = Increases Churn, Green = Reduces Churn)"},
            )
            fig_shap.update_traces(marker_color=bar_colors)
            fig_shap.update_layout(
                plot_bgcolor=paper_bg,
                paper_bgcolor=paper_bg,
                margin=dict(l=10, r=10, t=5, b=10),
                height=250,
                font=dict(family="Inter", color=text_color, size=11),
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color),
            )
            st.plotly_chart(fig_shap, use_container_width=True)


# =============================================================
# TAB 2: BATCH CSV SCORING
# =============================================================
with tab_batch:
    # 3 Pre-Loaded Demo CSV Datasets Box with Digits (No Emojis)
    st.markdown(
        inspect.cleandoc(f"""
        <div class="demo-preset-banner">
            <div class="demo-banner-title" style="color: {'#FFFFFF' if dark_mode_active else '#0F172A'};">Pre-Loaded Cohort Datasets</div>
            <div class="demo-banner-desc" style="color: {'#94A3B8' if dark_mode_active else '#64748B'};">Click any demo cohort below to execute automated risk scoring across dozens of customer records:</div>
        </div>
        """),
        unsafe_allow_html=True,
    )

    b1, b2, b3 = st.columns(3)
    batch_demo_selection = None

    with b1:
        if st.button("01 | VIP Champions (50 Accounts)", use_container_width=True):
            batch_demo_selection = "demo_1_high_value_vip.csv"
            st.session_state["loaded_batch_demo"] = batch_demo_selection
    with b2:
        if st.button("02 | At-Risk Buyers (50 Accounts)", use_container_width=True):
            batch_demo_selection = "demo_2_at_risk_customers.csv"
            st.session_state["loaded_batch_demo"] = batch_demo_selection
    with b3:
        if st.button("03 | Mixed Cohort (100 Accounts)", use_container_width=True):
            batch_demo_selection = "demo_3_mixed_cohort.csv"
            st.session_state["loaded_batch_demo"] = batch_demo_selection

    # Check if a demo batch was previously selected
    if batch_demo_selection is None and "loaded_batch_demo" in st.session_state:
        batch_demo_selection = st.session_state["loaded_batch_demo"]

    with card("Upload Customer CSV or Download Demo Templates", "Support format: CSV with Recency, Frequency, Monetary, Tenure, AvgOrderValue, UniqueProducts, IsUK:"):
        c_up, c_down = st.columns([2, 1], gap="medium")
        with c_up:
            uploaded_file = st.file_uploader("Upload CSV containing customer features", type=["csv"], key="batch_uploader")
            if uploaded_file is not None:
                batch_demo_selection = None
                if "loaded_batch_demo" in st.session_state:
                    del st.session_state["loaded_batch_demo"]

        with c_down:
            st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:{'#FFFFFF' if dark_mode_active else '#0F172A'}; margin-bottom:6px;'>Download Sample CSVs:</div>", unsafe_allow_html=True)
            for fname, label in [
                ("demo_1_high_value_vip.csv", "01 | VIP Champions CSV"),
                ("demo_2_at_risk_customers.csv", "02 | At-Risk Accounts CSV"),
                ("demo_3_mixed_cohort.csv", "03 | Mixed Cohort (100) CSV"),
            ]:
                fpath = DEMO_DIR / fname
                if fpath.exists():
                    with open(fpath, "rb") as f:
                        st.download_button(
                            label=label,
                            data=f.read(),
                            file_name=fname,
                            mime="text/csv",
                            use_container_width=True,
                            key=f"dl_{fname}",
                        )

    # Determine data source: uploaded file or selected demo
    batch_raw = None
    source_name = ""

    if uploaded_file is not None:
        try:
            batch_raw = pd.read_csv(uploaded_file)
            source_name = f"Uploaded File: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Error reading uploaded CSV: {e}")
    elif batch_demo_selection is not None:
        fpath = DEMO_DIR / batch_demo_selection
        if fpath.exists():
            batch_raw = pd.read_csv(fpath)
            clean_demo_name = batch_demo_selection.replace(".csv", "").replace("_", " ").title()
            source_name = f"Demo Cohort: {clean_demo_name}"

    if batch_raw is not None:
        req_cols = ["Recency", "Frequency", "Monetary", "Tenure", "AvgOrderValue", "UniqueProducts", "IsUK"]
        missing = [c for c in req_cols if c not in batch_raw.columns]

        if missing:
            st.error(f"Missing required columns in CSV: {missing}")
        else:
            with st.spinner("Calculating batch predictions with XGBoost..."):
                probs = models["xgb_churn"].predict_proba(batch_raw[req_cols])[:, 1]
                risks = [get_risk_category(p) for p in probs]
                ltv_logs = models["xgb_ltv"].predict(batch_raw[req_cols])
                ltvs = np.expm1(np.maximum(0, ltv_logs))

                results_df = batch_raw.copy()
                results_df["ChurnProbability"] = np.round(probs * 100, 1)
                results_df["RiskLevel"] = risks
                results_df["Predicted90DayLTV"] = np.round(ltvs, 2)

            st.success(f"Successfully processed {len(results_df):,} customer records ({source_name}).")

            # Batch KPI summary cards
            b_c1, b_c2, b_c3, b_c4 = st.columns(4)
            with b_c1:
                kpi_card("Batch Analyzed", f"{len(results_df):,}", "Accounts scored")
            with b_c2:
                avg_prob = np.mean(probs) * 100
                kpi_card("Avg Churn Risk", f"{avg_prob:.1f}%", "Cohort average")
            with b_c3:
                total_proj_ltv = np.sum(ltvs)
                kpi_card("Cohort 90-Day LTV", f"£{total_proj_ltv:,.2f}", "Expected forward spend")
            with b_c4:
                high_risks = (results_df["RiskLevel"] == "High").sum()
                kpi_card("High-Risk Accounts", f"{high_risks:,}", f"{(high_risks/len(results_df))*100:.1f}% of upload")

            with card("Scored Customer Results", "Interactive table of predictions with risk level mapping:"):
                st.dataframe(
                    results_df.style.map(
                        lambda val: "color: #EF4444; font-weight: 700;" if val == "High"
                        else ("color: #F59E0B; font-weight: 700;" if val == "Medium"
                        else "color: #10B981; font-weight: 700;"),
                        subset=["RiskLevel"]
                    ),
                    use_container_width=True,
                    height=350,
                )

                out_csv = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download Full Scored Results CSV",
                    data=out_csv,
                    file_name="retainiq_scored_predictions.csv",
                    mime="text/csv",
                    type="primary",
                )


# =============================================================
# TAB 3: EXECUTIVE OVERVIEW
# =============================================================
with tab_overview:
    total_cust = len(df_customers)
    overall_churn = df_customers["Churn"].mean() * 100
    avg_ltv = df_customers["FutureLTV"].mean()
    high_risk_count = (df_customers["Recency"] > 90).sum()
    high_risk_pct = (high_risk_count / total_cust) * 100

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi_card("Total Customers", f"{total_cust:,}", "Tracked in active cohort")
    with c2:
        kpi_card("90-Day Churn Rate", f"{overall_churn:.1f}%", "Historical baseline churn")
    with c3:
        kpi_card("Avg Forward LTV", f"£{avg_ltv:.2f}", "Projected spend per buyer")
    with c4:
        kpi_card("At-Risk Accounts", f"{high_risk_count:,}", f"{high_risk_pct:.1f}% of cohort")

    col_left, col_right = st.columns([1, 1], gap="medium")

    with col_left:
        with card("Churn Rate Across Customer Segments", "RFM-stratified churn propensities:"):
            seg_df = df_customers.groupby("CustomerSegment")["Churn"].agg(["mean", "count"]).reset_index()
            seg_df["ChurnRate"] = seg_df["mean"] * 100
            seg_df = seg_df.sort_values(by="ChurnRate", ascending=True)

            fig_bar = px.bar(
                seg_df,
                x="ChurnRate",
                y="CustomerSegment",
                orientation="h",
                text="ChurnRate",
                color="ChurnRate",
                color_continuous_scale=["#059669", "#D97706", "#DC2626"],
                labels={"ChurnRate": "Churn Rate (%)", "CustomerSegment": "RFM Segment"},
            )
            fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig_bar.update_layout(
                plot_bgcolor=paper_bg,
                paper_bgcolor=paper_bg,
                margin=dict(l=10, r=20, t=10, b=10),
                height=300,
                coloraxis_showscale=False,
                font=dict(family="Inter", color=text_color),
                xaxis=dict(gridcolor=grid_color),
                yaxis=dict(gridcolor=grid_color),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col_right:
        with card("Cohort Risk Distribution", "Classification based on recency & activity status:"):
            risk_labels = []
            for r in df_customers["Recency"]:
                if r < 30:
                    risk_labels.append("Low")
                elif r < 75:
                    risk_labels.append("Medium")
                else:
                    risk_labels.append("High")
            risk_counts = pd.Series(risk_labels).value_counts().reset_index()
            risk_counts.columns = ["RiskLevel", "Count"]

            fig_donut = px.pie(
                risk_counts,
                names="RiskLevel",
                values="Count",
                hole=0.6,
                color="RiskLevel",
                color_discrete_map={"Low": "#059669", "Medium": "#D97706", "High": "#DC2626"},
            )
            fig_donut.update_traces(textinfo="percent+label")
            fig_donut.update_layout(
                plot_bgcolor=paper_bg,
                paper_bgcolor=paper_bg,
                margin=dict(l=10, r=10, t=10, b=10),
                height=300,
                font=dict(family="Inter", color=text_color),
                showlegend=False,
            )
            st.plotly_chart(fig_donut, use_container_width=True)

    with card("Kaplan-Meier Survival Curves by Segment", "Empirical retention rate over 90-day observation window:"):
        fig_km = go.Figure()
        kmf_dict = models["kmf_dict"] if models else {}
        palette = {
            "Champions": "#059669",
            "Loyal Customers": "#635BFF",
            "Potential Loyalists": "#0284C7",
            "At Risk": "#D97706",
            "Hibernating": "#DC2626",
            "Global": "#94A3B8" if dark_mode_active else "#0F172A",
        }
        for seg_name, kmf in kmf_dict.items():
            timeline = kmf.timeline
            survival = kmf.survival_function_[kmf._label]
            color = palette.get(seg_name, "#635BFF")
            fig_km.add_trace(
                go.Scatter(
                    x=timeline,
                    y=survival,
                    mode="lines",
                    name=seg_name,
                    line=dict(color=color, width=3 if seg_name == "Global" else 2, dash="dash" if seg_name == "Global" else "solid"),
                )
            )

        fig_km.update_layout(
            plot_bgcolor=paper_bg,
            paper_bgcolor=paper_bg,
            margin=dict(l=20, r=20, t=20, b=20),
            height=340,
            xaxis=dict(title="Days After Cutoff Window", gridcolor=grid_color),
            yaxis=dict(title="Probability of Retaining Activity", range=[0, 1.05], gridcolor=grid_color),
            font=dict(family="Inter", color=text_color),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_km, use_container_width=True)


# =============================================================
# TAB 4: MODEL INSIGHTS & BENCHMARKS
# =============================================================
with tab_insights:
    with card("Model Architecture Benchmarks (Holdout Test N=1,313)", "Rigorous comparison against linear baseline:"):
        lr_m = metrics_data.get("churn_classification", {}).get("logistic_regression", {})
        xgb_m = metrics_data.get("churn_classification", {}).get("xgboost", {})

        bench_df = pd.DataFrame({
            "Evaluation Metric": ["ROC-AUC Score", "F1-Score", "Recall (Sensitivity)", "Precision", "Accuracy"],
            "Logistic Regression Baseline": [
                f"{lr_m.get('roc_auc', 0.7796):.4f}",
                f"{lr_m.get('f1_score', 0.7340):.4f}",
                f"{lr_m.get('recall', 0.6955):.4f}",
                f"{lr_m.get('precision', 0.7771):.4f}",
                f"{lr_m.get('accuracy', 0.7113):.4f}",
            ],
            "XGBoost Classifier (Deployed)": [
                f"{xgb_m.get('roc_auc', 0.7854):.4f}",
                f"{xgb_m.get('f1_score', 0.7559):.4f}",
                f"{xgb_m.get('recall', 0.7513):.4f}",
                f"{xgb_m.get('precision', 0.7604):.4f}",
                f"{xgb_m.get('accuracy', 0.7220):.4f}",
            ],
            "Operational Advantage": ["+0.0058", "+0.0219", "+0.0558 (+5.58% Churn Detection)", "-0.0167", "+0.0107"],
        })
        st.table(bench_df)
        st.caption("Key Takeaway: In customer churn prevention, Recall is the primary business metric. XGBoost detects 565 out of 752 actual churners (+5.58% higher than linear models).")

    col_m1, col_m2 = st.columns(2, gap="medium")

    with col_m1:
        with card("Confusion Matrix Heatmap", "XGBoost holdout test evaluation:"):
            cm_xgb = np.array(xgb_m.get("confusion_matrix", [[383, 178], [187, 565]]))
            fig_cm = go.Figure(
                data=go.Heatmap(
                    z=cm_xgb,
                    x=["Retained (Pred)", "Churned (Pred)"],
                    y=["Retained (Actual)", "Churned (Actual)"],
                    text=cm_xgb,
                    texttemplate="%{text}",
                    colorscale="Purples",
                    showscale=False,
                )
            )
            fig_cm.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=10, b=10),
                paper_bgcolor=paper_bg,
                font=dict(family="Inter", color=text_color),
            )
            st.plotly_chart(fig_cm, use_container_width=True)

    with col_m2:
        with card("ROC Curve Comparison", "Receiver Operating Characteristic:"):
            test_data_path = MODELS_DIR / "test_data.joblib"
            if test_data_path.exists():
                from sklearn.metrics import roc_curve
                td = joblib.load(test_data_path)
                y_true = td["y_churn_test"]
                xgb_probs = models["xgb_churn"].predict_proba(td["X_test"])[:, 1]
                lr_probs = models["lr"].predict_proba(td["X_test"])[:, 1]

                fpr_xgb, tpr_xgb, _ = roc_curve(y_true, xgb_probs)
                fpr_lr, tpr_lr, _ = roc_curve(y_true, lr_probs)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, name=f"Logistic Regression (AUC: {lr_m.get('roc_auc', 0.779):.3f})", line=dict(color="#64748B", dash="dash")))
                fig_roc.add_trace(go.Scatter(x=fpr_xgb, y=tpr_xgb, name=f"XGBoost (AUC: {xgb_m.get('roc_auc', 0.785):.3f})", line=dict(color="#818CF8", width=2.5)))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], name="Random Baseline", line=dict(color="#475569" if dark_mode_active else "#CBD5E1", dash="dot")))

                fig_roc.update_layout(
                    height=280,
                    margin=dict(l=20, r=20, t=10, b=10),
                    paper_bgcolor=paper_bg,
                    plot_bgcolor=paper_bg,
                    xaxis=dict(title="False Positive Rate", gridcolor=grid_color),
                    yaxis=dict(title="True Positive Rate", gridcolor=grid_color),
                    font=dict(family="Inter", color=text_color),
                    legend=dict(orientation="h", y=-0.25),
                )
                st.plotly_chart(fig_roc, use_container_width=True)

    # Global SHAP Card
    with card("Global Feature Importance (SHAP)", "What factors most heavily govern customer churn across all accounts:"):
        shap_img = ROOT_DIR / "reports/figures/08_shap_summary.png"
        if shap_img.exists():
            st.image(str(shap_img), use_container_width=True, caption="Global SHAP Summary: Recency (days since last purchase) is the single most predictive factor of churn propensity.")

# Footer without author/student attribution and without emojis
render_footer()
