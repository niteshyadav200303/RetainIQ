"""
UI components and layout helpers for RetainIQ.
Includes:
- kpi_card: Metric display card with title, value, and subtitle
- risk_badge: Low / Medium / High colored pill badge
- card: Context manager for card containers
- inject_custom_css: Loads style.css into Streamlit
"""

from contextlib import contextmanager
from pathlib import Path
import streamlit as st


def inject_custom_css():
    """Load and inject modern Stripe/Mixpanel custom CSS."""
    css_path = Path(__file__).resolve().parent / "style.css"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)


def kpi_card(title: str, value: str, subtitle: str = ""):
    """Render a modern SaaS KPI card."""
    html = f"""
    <div class="kpi-container">
        <div class="kpi-title">{title}</div>
        <div class="kpi-value">{value}</div>
        <div class="kpi-subtitle">{subtitle}</div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    """Return HTML string for colored risk pill badge."""
    lvl = level.lower().strip()
    if lvl == "low":
        cls = "risk-low"
        label = "Low Risk"
    elif lvl in ("medium", "med"):
        cls = "risk-medium"
        label = "Medium Risk"
    else:
        cls = "risk-high"
        label = "High Risk"

    return f'<span class="risk-pill {cls}">{label}</span>'


@contextmanager
def card(title: str = None):
    """
    Context manager that wraps contents inside a clean white SaaS card container.
    Usage:
        with card("Customer Overview"):
            st.write("...")
    """
    header_html = f'<div class="card-header-title">{title}</div>' if title else ""
    st.markdown(f'<div class="retain-card">{header_html}', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def render_footer():
    """Render project footer."""
    st.markdown(
        """
        <div class="custom-footer">
            Built by <strong>Nitesh</strong> | Final Year Mini Project, AI & DS II &bull; RetainIQ &copy; 2026
        </div>
        """,
        unsafe_allow_html=True,
    )
