"""
UI components and layout helpers for RetainIQ.
Includes:
- kpi_card: Metric display card with title, value, and subtitle
- risk_badge: Low / Medium / High colored pill badge (no emojis)
- card: Context manager for clean SaaS card containers
- recommendation_card: Business action & retention strategy based on risk (no emojis)
- inject_custom_css: Loads style.css into Streamlit with Dark/Light mode tokens
- render_header: Polished SaaS banner
- render_footer: Enterprise footer (no emojis, no author mention)
"""

from contextlib import contextmanager
import inspect
from pathlib import Path
import streamlit as st


def inject_custom_css(dark_mode: bool = False):
    """Load and inject modern custom CSS supporting both Light and Dark/Black themes."""
    css_path = Path(__file__).resolve().parent / "style.css"
    theme_class = "dark-theme" if dark_mode else "light-theme"
    if css_path.exists():
        with open(css_path, "r", encoding="utf-8") as f:
            css_content = f.read()
        st.markdown(f"<style>{css_content}</style><div class='{theme_class}' style='display:none;'></div>", unsafe_allow_html=True)


def kpi_card(title: str, value: str, subtitle: str = "", badge: str = None):
    """Render a modern SaaS KPI card with optional trend/status badge."""
    badge_html = f'<span class="kpi-badge">{badge}</span>' if badge else ""
    html = f'<div class="kpi-container"><div class="kpi-header-row"><div class="kpi-title">{title}</div>{badge_html}</div><div class="kpi-value">{value}</div><div class="kpi-subtitle">{subtitle}</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def risk_badge(level: str) -> str:
    """Return HTML string for colored risk pill badge without emojis."""
    lvl = level.lower().strip()
    if lvl == "low":
        cls = "risk-low"
        label = "Low Churn Risk"
    elif lvl in ("medium", "med"):
        cls = "risk-medium"
        label = "Medium Churn Risk"
    else:
        cls = "risk-high"
        label = "High Churn Risk"

    return f'<span class="risk-pill {cls}">{label}</span>'


@contextmanager
def card(title: str = None, subtitle: str = None):
    """
    Context manager that wraps contents inside a clean SaaS card container.
    """
    header_html = ""
    if title:
        sub_html = f'<div class="card-header-subtitle">{subtitle}</div>' if subtitle else ""
        header_html = f'<div class="card-header-title">{title}</div>{sub_html}'
    
    st.markdown(f'<div class="retain-card">{header_html}', unsafe_allow_html=True)
    try:
        yield
    finally:
        st.markdown("</div>", unsafe_allow_html=True)


def recommendation_card(risk_level: str, ltv: float):
    """Render plain-English actionable retention recommendations without emojis."""
    lvl = risk_level.lower().strip()
    if lvl == "low":
        title = "Strategic Action: Retain and Elevate (VIP Tier)"
        color_class = "rec-low"
        actions = [
            "<strong>VIP Loyalty Program:</strong> Enroll customer into exclusive VIP perks and early access promotions.",
            "<strong>Cross-Sell Synergy:</strong> Target with complementary collections to maximize basket size.",
            "<strong>Brand Advocacy:</strong> Encourage review submissions with automated milestone credits."
        ]
        summary = f"Customer has low churn propensity and strong projected forward value (<strong>£{ltv:.2f}</strong>). Focus on long-term retention."
    elif lvl in ("medium", "med"):
        title = "Strategic Action: Proactive Re-Engagement"
        color_class = "rec-medium"
        actions = [
            "<strong>Timely Outreach:</strong> Send a personalized re-engagement campaign within 7 days.",
            "<strong>Incentive Offer:</strong> Include a time-sensitive 15% promotional voucher or free shipping.",
            "<strong>Satisfaction Check:</strong> Trigger a brief satisfaction survey to eliminate buying friction."
        ]
        summary = f"Customer activity is cooling down with expected forward spend around <strong>£{ltv:.2f}</strong>. Proactive outreach prevents churn."
    else:
        title = "Strategic Action: Win-Back Campaign or Budget Preservation"
        color_class = "rec-high"
        actions = [
            "<strong>Targeted Win-Back:</strong> Deliver high-impact discount (e.g., 25% off their top historical category).",
            "<strong>Exit Feedback:</strong> Dispatch automated feedback questionnaire to evaluate drop-off reasons.",
            "<strong>Cost Optimization:</strong> Avoid expensive paid ad re-targeting if acquisition costs exceed projected LTV."
        ]
        summary = f"High churn probability with low projected spend (<strong>£{ltv:.2f}</strong>). Prioritize high-margin win-backs or conserve marketing budget."

    bullets = "".join([f"<li style='margin-bottom: 6px;'>{a}</li>" for a in actions])
    html = inspect.cleandoc(f"""
    <div class="recommendation-box {color_class}">
        <div class="rec-title">{title}</div>
        <div class="rec-summary">{summary}</div>
        <ul class="rec-list">
            {bullets}
        </ul>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def render_header(title: str, subtitle: str):
    """Render page top title banner."""
    html = inspect.cleandoc(f"""
    <div class="app-header">
        <h1 class="app-title">{title}</h1>
        <div class="app-subtitle">{subtitle}</div>
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)


def render_footer():
    """Render enterprise SaaS footer without author name and without emojis."""
    html = inspect.cleandoc("""
    <div class="custom-footer">
        RetainIQ &bull; Customer Churn & Lifetime Value AI Engine &bull; Enterprise Intelligence Platform &copy; 2026
    </div>
    """)
    st.markdown(html, unsafe_allow_html=True)
