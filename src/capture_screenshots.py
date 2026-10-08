"""
Take high-resolution screenshots of all 4 pages in the RetainIQ Streamlit app.
"""

import time
from pathlib import Path
from playwright.sync_api import sync_playwright

FIGURES_DIR = Path("reports/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def capture_all_pages():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 900})
        page = context.new_page()

        print("[Screenshot] Navigating to http://localhost:8501...")
        page.goto("http://localhost:8501", wait_until="networkidle", timeout=60000)
        time.sleep(4)

        # 1. Overview
        print("[Screenshot] Capturing 1. Overview...")
        page.screenshot(path=str(FIGURES_DIR / "10_app_overview.png"), full_page=True)

        # 2. Single Prediction
        print("[Screenshot] Clicking Single Prediction...")
        page.locator("text=Single Prediction").first.click()
        time.sleep(3)
        # Click Predict button
        btn = page.locator("button:has-text('Generate RetainIQ Prediction')")
        if btn.count() > 0:
            btn.first.click()
            time.sleep(2)
        print("[Screenshot] Capturing 2. Single Prediction...")
        page.screenshot(path=str(FIGURES_DIR / "11_app_single_prediction.png"), full_page=True)

        # 3. Batch Prediction
        print("[Screenshot] Clicking Batch Prediction...")
        page.locator("text=Batch Prediction").first.click()
        time.sleep(3)
        print("[Screenshot] Capturing 3. Batch Prediction...")
        page.screenshot(path=str(FIGURES_DIR / "12_app_batch_prediction.png"), full_page=True)

        # 4. Model Insights
        print("[Screenshot] Clicking Model Insights...")
        page.locator("text=Model Insights").first.click()
        time.sleep(3)
        print("[Screenshot] Capturing 4. Model Insights...")
        page.screenshot(path=str(FIGURES_DIR / "13_app_model_insights.png"), full_page=True)

        browser.close()
        print("[Screenshot] All 4 page screenshots successfully saved to reports/figures/!")


if __name__ == "__main__":
    capture_all_pages()
