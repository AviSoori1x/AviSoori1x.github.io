#!/usr/bin/env python3
"""Render og.png, the 1200x630 social card, by screenshotting an inline card.

LinkedIn and X will not show a preview image at all without one, and the first
share of the Shieldstral guide went out bare because of exactly that.

    PLAYWRIGHT_BROWSERS_PATH=/mnt/vast/home/avi/.cache/ms-playwright \
      /tmp/srenv/bin/python3 make_og.py
"""
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
data = json.loads((HERE / "robostral_data.json").read_text(encoding="utf-8"))
h = data["headline"]

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
  h1 {{ font-size:82px; line-height:.98; letter-spacing:-.035em; color:#141922;
    font-weight:700; margin-top:26px; max-width:760px; }}
  h1 em {{ font-style:normal; color:#b4551a; }}
  p {{ font:400 23px/1.42 'JetBrains Mono',monospace; color:#39404f;
    max-width:680px; margin-top:26px; }}
  .row {{ display:flex; gap:12px; align-items:center; }}
  .chip {{ font:500 19px/1 'JetBrains Mono',monospace; color:#1c6e8c;
    border:1.5px solid rgba(28,110,140,.5); background:rgba(28,110,140,.08);
    padding:12px 18px; border-radius:999px; }}
  .chip b {{ color:#141922; font-weight:700; }}
  .art {{ position:absolute; right:56px; top:120px; }}
</style></head><body>
<div class="grid"></div>
<svg class="art" width="400" height="330" viewBox="0 0 400 330">
  <defs><clipPath id="cv"><rect x="0" y="0" width="400" height="256" rx="14"/></clipPath></defs>
  <g clip-path="url(#cv)">
    <rect x="0" y="0" width="400" height="256" fill="#ece7dc"/>
    <!-- a corridor in one-point perspective, vanishing at the waypoint -->
    <path d="M0 256 L150 132 L250 132 L400 256 Z" fill="#e4ded1"/>
    <path d="M0 0 L150 132 L150 256 L0 256 Z" fill="#d8d1c1"/>
    <path d="M400 0 L250 132 L250 256 L400 256 Z" fill="#d8d1c1"/>
    <rect x="0" y="0" width="400" height="132" fill="#efeade"/>
    <g stroke="#c9c1b2" stroke-width="1.4" fill="none">
      <path d="M0 300 L162 140"/><path d="M400 300 L238 140"/>
      <path d="M60 256 L172 136"/><path d="M340 256 L228 136"/>
    </g>
    <rect x="176" y="120" width="48" height="16" rx="3" fill="#b4551a"/>
    <path d="M150 132 L250 132" stroke="#c9c1b2" stroke-width="1.4"/>
    <path d="M200 256 C 198 210, 202 178, 200 140" stroke="#1c6e8c"
      stroke-width="7" fill="none" stroke-linecap="round"/>
  </g>
  <rect x="0" y="0" width="400" height="256" rx="14" fill="none" stroke="rgba(31,37,48,.22)"/>
  <g>
    <circle cx="200" cy="128" r="30" fill="none" stroke="#f4f1ea" stroke-width="8"/>
    <circle cx="200" cy="128" r="30" fill="none" stroke="#a32d2d" stroke-width="4"/>
    <g stroke="#f4f1ea" stroke-width="8">
      <path d="M158 128 h14"/><path d="M228 128 h14"/>
      <path d="M200 86 v14"/><path d="M200 156 v14"/></g>
    <g stroke="#a32d2d" stroke-width="4">
      <path d="M158 128 h14"/><path d="M228 128 h14"/>
      <path d="M200 86 v14"/><path d="M200 156 v14"/></g>
  </g>
  <text x="200" y="300" text-anchor="middle" font-family="'JetBrains Mono',monospace"
    font-size="19" fill="#a32d2d" font-weight="500">u 0.500   v 0.407</text>
</svg>
<div class="wrap">
  <div>
    <div class="eyebrow">a visual guide</div>
    <h1>Robostral&nbsp;Navigate,<br>taken <em>apart</em></h1>
    <p>One RGB camera. It navigates by pointing at a pixel.</p>
  </div>
  <div class="row">
    <span class="chip">arXiv <b>2607.20785</b></span>
    <span class="chip">R2R-CE SR <b>{h['r2r_sr']}</b></span>
    <span class="chip">RxR-CE SR <b>{h['rxr_sr']}</b></span>
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
