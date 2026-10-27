"""Animated tour of the dashboard from the tab screenshots (docs/tour.gif)."""
from pathlib import Path

from PIL import Image

D = Path(__file__).resolve().parent.parent / "docs"
frames = []
for name in ["overview", "live", "channel", "ranging", "navigation", "real"]:
    im = Image.open(D / "screenshots" / f"{name}.png").convert("RGB")
    im = im.crop((40, 60, 1400, 60 + 1050))
    im.thumbnail((960, 960))
    frames.append(im)
seq, dur = [], []
for i, f in enumerate(frames):
    seq.append(f)
    dur.append(2300)
    nxt = frames[(i + 1) % len(frames)]
    for a in (0.33, 0.66):
        seq.append(Image.blend(f, nxt.resize(f.size), a))
        dur.append(90)
pal = [s.quantize(colors=128, method=Image.Quantize.MEDIANCUT) for s in seq]
pal[0].save(D / "tour.gif", save_all=True, append_images=pal[1:], duration=dur, loop=0, optimize=True)
print((D / "tour.gif").stat().st_size // 1024, "KB")
