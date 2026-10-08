from playwright.sync_api import sync_playwright
from pathlib import Path

def verify_sidebar_fix():
    FIGURES = Path("reports/figures")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True)
        page = b.new_page(viewport={"width": 1440, "height": 900})
        page.goto("http://localhost:8501")
        page.wait_for_timeout(4000)

        # Collapse the sidebar
        collapse_btn = page.locator('button[data-testid="stBaseButton-headerNoPadding"]')
        print(f"Collapse button count: {collapse_btn.count()}")
        if collapse_btn.count() > 0:
            collapse_btn.first.click()
            page.wait_for_timeout(1200)

        # Check expand button
        expand_btn = page.locator('button[data-testid="stExpandSidebarButton"]')
        bb = expand_btn.bounding_box()
        is_visible = expand_btn.count() > 0 and bb is not None
        print(f"Expand button visible: {is_visible}, bounding box: {bb}")

        # Screenshot collapsed state
        page.screenshot(path=str(FIGURES / "14_sidebar_collapsed.png"))
        print("Captured collapsed screenshot.")

        # Click expand button to reopen
        if is_visible:
            expand_btn.first.click()
            page.wait_for_timeout(1200)
            page.screenshot(path=str(FIGURES / "15_sidebar_expanded.png"))
            print("Captured expanded screenshot after re-opening.")

        b.close()

if __name__ == "__main__":
    verify_sidebar_fix()
