import os, time
from playwright.sync_api import sync_playwright

here = os.path.dirname(__file__)
html_path = os.path.abspath(os.path.join(here, "can_ids_demo.html"))
out_dir = os.path.join(here, "..", "results")
os.makedirs(out_dir, exist_ok=True)

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(viewport={"width": 900, "height": 700})
    page.goto(f"file://{html_path}")

    time.sleep(0.3)
    page.screenshot(path=os.path.join(out_dir, "demo_normal.png"))
    print("저장:", os.path.join(out_dir, "demo_normal.png"))

    for _ in range(20):
        time.sleep(0.4)
        if page.locator(".dot.alert").count() > 0:
            page.screenshot(path=os.path.join(out_dir, "demo_attack_detected.png"))
            print("저장:", os.path.join(out_dir, "demo_attack_detected.png"))
            break

    browser.close()
