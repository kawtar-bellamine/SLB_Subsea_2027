"""Screenshots of every dashboard tab (needs the app running on port 8655)."""
import sys
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent.parent / "docs" / "screenshots"
OUT.mkdir(parents=True, exist_ok=True)
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8655"
NAMES = ["overview", "live", "channel", "ranging", "navigation", "real", "method"]
errors = []
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_page(viewport={"width": 1440, "height": 1700}, device_scale_factor=1)
    pg.on("pageerror", lambda e: errors.append(str(e)))
    pg.goto(URL, wait_until="networkidle")
    pg.wait_for_selector("[data-testid=stTab]", timeout=120_000)
    pg.wait_for_timeout(4000)
    tabs = pg.locator("[data-testid=stTab]")
    for i, name in enumerate(NAMES):
        tabs.nth(i).click()
        pg.wait_for_timeout(1500)
        for _ in range(60):
            if pg.locator("[data-testid=stSpinner]").count() == 0:
                break
            pg.wait_for_timeout(1000)
        pg.wait_for_timeout(2500)
        exc = pg.locator("[data-testid=stException]").count()
        pg.screenshot(path=str(OUT / f"{name}.png"), full_page=True)
        print(name, "exceptions:", exc)
    b.close()
print("page errors:", errors)
