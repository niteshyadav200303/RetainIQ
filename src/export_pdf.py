"""
Export reports/report_draft.md to reports/report.pdf using markdown-it and Playwright.
"""

from pathlib import Path
import markdown_it
from playwright.sync_api import sync_playwright

MD_PATH = Path("reports/report_draft.md")
PDF_PATH = Path("reports/report.pdf")


def export_markdown_to_pdf():
    text = MD_PATH.read_text(encoding="utf-8")
    md = markdown_it.MarkdownIt()
    html_content = md.render(text)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        margin: 30px 40px;
        line-height: 1.65;
        color: #0A2540;
        font-size: 11pt;
    }}
    h1 {{
        font-size: 22pt;
        font-weight: 700;
        color: #0A2540;
        border-bottom: 2px solid #635BFF;
        padding-bottom: 8px;
        margin-top: 10px;
    }}
    h2 {{
        font-size: 15pt;
        font-weight: 600;
        color: #0A2540;
        margin-top: 24px;
        border-bottom: 1px solid #E3E8EE;
        padding-bottom: 4px;
    }}
    h3 {{
        font-size: 12pt;
        font-weight: 600;
        color: #2E384D;
        margin-top: 16px;
    }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 18px 0;
        font-size: 10pt;
    }}
    th, td {{
        border: 1px solid #E3E8EE;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #F6F9FC;
        font-weight: 600;
        color: #0A2540;
    }}
    code {{
        background-color: #F6F9FC;
        border: 1px solid #E3E8EE;
        padding: 2px 5px;
        border-radius: 4px;
        font-size: 9.5pt;
    }}
    hr {{
        border: none;
        border-top: 1px solid #E3E8EE;
        margin: 24px 0;
    }}
    blockquote {{
        border-left: 4px solid #635BFF;
        margin: 0;
        padding-left: 14px;
        color: #6B7C93;
    }}
</style>
</head>
<body>
{html_content}
</body>
</html>
"""

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content(full_html)
        page.pdf(
            path=str(PDF_PATH),
            format="A4",
            margin={"top": "15mm", "bottom": "15mm", "left": "15mm", "right": "15mm"},
            print_background=True,
        )
        browser.close()

    print(f"[PDF] Successfully created {PDF_PATH} ({PDF_PATH.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    export_markdown_to_pdf()
