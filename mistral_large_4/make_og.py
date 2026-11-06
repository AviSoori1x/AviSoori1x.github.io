#!/usr/bin/env python3
"""Render og.png, the 1200x630 social card, by screenshotting an inline card.

LinkedIn and X will not show a preview image at all without one, and the first
share of the Shieldstral guide went out bare because of exactly that.

    PLAYWRIGHT_BROWSERS_PATH=/mnt/vast/home/avi/.cache/ms-playwright \
      /mnt/vast/home/avi/science/website/.venv/bin/python make_og.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
raw = json.loads((HERE / "ml4_data.json").read_text(encoding="utf-8"))["facts"]
# JSON gives 1.0 and 606.0; print whole numbers without the trailing zero
f = {k: (int(v) if isinstance(v, float) and v.is_integer() else v) for k, v in raw.items()}


def stack(x0, after):
    """Twelve layers as context bars. Before surgery three of every four look
    back only a short window; after it, every layer sees the whole context."""
    rows = []
    for i in range(12):
        y = 44 + i * 26
        rows.append(f'<rect x="{x0}" y="{y}" width="176" height="16" rx="5" fill="#e9e3d6"/>')
        if after or i % 4 == 3:
            rows.append(f'<rect x="{x0}" y="{y}" width="176" height="16" rx="5" fill="#b4551a"/>')
        else:
            rows.append(f'<rect x="{x0 + 140}" y="{y}" width="36" height="16" rx="5" fill="#8a909c"/>')
    return "\n  ".join(rows)


CARD = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@400;700&family=JetBrains+Mono:wght@400;500&display=swap">
<style>
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ width:1200px; height:630px; background:#f4f1ea; overflow:hidden;
    font-family:'Sora',sans-serif; position:relative; }}
  .grid {{ position:absolute; inset:0;
    background-image:radial-gradient(rgba(31,37,48,.13) 1px, transparent 1px);
    background-size:28px 28px; }}
  .wrap {{ position:relative; padding:64px 70px; height:100%;
    display:flex; flex-direction:column; justify-content:space-between; }}
  .eyebrow {{ font:500 17px/1 'JetBrains Mono',monospace; letter-spacing:.20em;
    text-transform:uppercase; color:#b4551a; }}
  h1 {{ font-size:74px; line-height:.98; letter-spacing:-.035em; color:#141922;
    font-weight:700; margin-top:26px; white-space:nowrap; }}
  h1 em {{ font-style:normal; color:#b4551a; }}
  p {{ font:400 22px/1.42 'JetBrains Mono',monospace; color:#39404f;
    max-width:600px; margin-top:24px; }}
  .row {{ display:flex; gap:12px; align-items:center; }}
  .chip {{ font:500 19px/1 'JetBrains Mono',monospace; color:#1c6e8c;
    border:1.5px solid rgba(28,110,140,.5); background:rgba(28,110,140,.08);
    padding:12px 18px; border-radius:999px; }}
  .chip b {{ color:#141922; font-weight:700; }}
  .art {{ position:absolute; right:56px; top:150px; }}
  .art text {{ font:500 16px 'JetBrains Mono',monospace; fill:#39404f; }}
</style></head><body>
<div class="grid"></div>
<svg class="art" width="440" height="372" viewBox="0 0 440 372">
  <text x="0" y="26">before</text>
  <text x="264" y="26">after surgery</text>
  {stack(0, False)}
  {stack(264, True)}
  <path d="M196 196 H244" stroke="#39404f" stroke-width="3" stroke-linecap="round"/>
  <path d="M234 186 L246 196 L234 206" stroke="#39404f" stroke-width="3" fill="none"
    stroke-linecap="round" stroke-linejoin="round"/>
</svg>
<div class="wrap">
  <div>
    <div class="eyebrow">a visual guide</div>
    <h1>Mistral Large 4,<br>taken <em>apart</em></h1>
    <p>Its cheap attention hurt agentic work, so it was operated on in the middle of training.</p>
  </div>
  <div class="row">
    <span class="chip"><b>{f['size.total_t']}T</b> params, <b>{f['size.active_b']}B</b> active</span>
    <span class="chip">FrontierCode 2.0 <b>{f['intro.frontiercode']}%</b></span>
    <span class="chip"><b>{f['q.gb']} GB</b> on {f['cp.gpus']}&times;GB300</span>
  </div>
</div>
</body></html>"""

tmp = HERE / "_og.html"
tmp.write_text(CARD, encoding="utf-8")

from playwright.sync_api import sync_playwright  # noqa: E402

with sync_playwright() as p:
    br = p.chromium.launch(args=["--no-sandbox"])
    pg = br.new_page(viewport={"width": 1200, "height": 630}, device_scale_factor=1)
    pg.goto(tmp.as_uri(), wait_until="networkidle")
    pg.wait_for_timeout(1200)
    pg.screenshot(path=str(HERE / "og.png"))
    br.close()
tmp.unlink()
print("wrote og.png", (HERE / "og.png").stat().st_size, "bytes")
