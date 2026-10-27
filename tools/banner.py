"""Banner (README) and LinkedIn visual, rendered from HTML with the numbers of results/summary.json."""
import base64
import json
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parent.parent
S = json.loads((ROOT / "results" / "summary.json").read_text())
nav = {r["method"]: r for r in S["navigation"]}
best = nav["EKF + calibration + gate + ML"]["rmse_m"] * 100
img = base64.b64encode((ROOT / "results" / "figures" / "fig3_constellations.png").read_bytes()).decode()

waves = "".join(
    f'<circle cx="0" cy="0" r="{r}" fill="none" stroke="rgba(63,208,255,{0.55 - r / 900:.2f})" stroke-width="2"/>'
    for r in range(40, 460, 42))


def html(w, h, big):
    return f"""<html><head><style>
body {{ margin:0; width:{w}px; height:{h}px; overflow:hidden; font-family: 'Inter','Segoe UI',Arial,sans-serif;
  background: radial-gradient(900px 500px at 85% 20%, #11507d 0%, #0a2a45 40%, #061726 75%); color:#e6f1ff; }}
.wrap {{ position:absolute; inset:0; padding:{48 if big else 40}px {56 if big else 48}px; }}
.kick {{ letter-spacing:4px; font-size:{14 if big else 13}px; color:#3fd0ff; font-weight:700; }}
h1 {{ font-size:{58 if big else 52}px; margin:10px 0 8px; letter-spacing:.5px; }}
.sub {{ font-size:{20 if big else 18}px; color:#8fb3d1; max-width:{660 if big else 700}px; line-height:1.4; }}
.k {{ position:absolute; left:{56 if big else 48}px; bottom:{48 if big else 36}px; display:flex; gap:14px; }}
.c {{ background:rgba(11,34,54,.85); border:1px solid #1d3a55; border-radius:14px; padding:12px 16px; min-width:120px; }}
.c b {{ display:block; font-size:{26 if big else 24}px; color:#3fd0ff; }}
.c span {{ font-size:12px; color:#8fb3d1; }}
svg {{ position:absolute; right:-40px; top:-40px; }}
.fig {{ position:absolute; right:{40 if big else 30}px; top:{h * 0.30:.0f}px; width:{w * 0.40:.0f}px;
  border-radius:12px; border:1px solid #1d3a55; opacity:.95; }}
</style></head><body>
<svg width="{w * 0.62:.0f}" height="{h + 80}" viewBox="-{w * 0.62:.0f} 0 {w * 0.62:.0f} {h + 80}">
<g transform="translate(-60,60)">{waves}</g></svg>
<img class="fig" src="data:image/png;base64,{img}">
<div class="wrap"><div class="kick">UNDERWATER ROBOTICS · SIGNAL PROCESSING</div>
<h1>Subsea Acoustic Link</h1>
<div class="sub">Acoustic communication and positioning for AUVs: multipath, Doppler and real hydrophone noise,
an adaptive equaliser, chirp ranging at the Cramér-Rao bound and LBL navigation.</div></div>
<div class="k">
<div class="c"><b>0 bit errors</b><span>from 18 dB, 1 km, 1 m/s</span></div>
<div class="c"><b>1.4 cm/s</b><span>Doppler speed error</span></div>
<div class="c"><b>at the bound</b><span>ranging vs Cramér-Rao, 20 dB</span></div>
<div class="c"><b>{best:.1f} cm</b><span>navigation RMSE</span></div>
</div></body></html>"""


with sync_playwright() as p:
    b = p.chromium.launch()
    for name, (w, h, big) in {"banner": (1600, 520, True), "linkedin": (1200, 627, False)}.items():
        pg = b.new_page(viewport={"width": w, "height": h}, device_scale_factor=1.5 if name == "banner" else 2)
        pg.set_content(html(w, h, big))
        pg.wait_for_timeout(500)
        pg.screenshot(path=str(ROOT / "docs" / f"{name}.png"))
    b.close()
print("ok")
