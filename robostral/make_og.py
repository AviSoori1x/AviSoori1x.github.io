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
  h1 {{ font-size:78px; line-height:.98; letter-spacing:-.035em; color:#141922;
    font-weight:700; margin-top:26px; max-width:690px; }}
  h1 em {{ font-style:normal; color:#b4551a; }}
  p {{ font:400 22px/1.42 'JetBrains Mono',monospace; color:#39404f;
    max-width:620px; margin-top:24px; }}
  .row {{ display:flex; gap:12px; align-items:center; }}
  .chip {{ font:500 19px/1 'JetBrains Mono',monospace; color:#1c6e8c;
    border:1.5px solid rgba(28,110,140,.5); background:rgba(28,110,140,.08);
    padding:12px 18px; border-radius:999px; }}
  .chip b {{ color:#141922; font-weight:700; }}
  .art {{ position:absolute; right:34px; top:116px; }}
</style></head><body>
<div class="grid"></div>
<svg class="art" width="470" height="404" viewBox="0 0 470 404">
  <!-- corridor floor, receding to a vanishing point up and right -->
  <path d="M20 388 L212 150 L318 150 L452 388 Z" fill="#e9e3d6"/>
  <g stroke="#d5cdbc" stroke-width="1.6" fill="none">
    <path d="M74 388 L226 156"/><path d="M400 388 L306 156"/>
    <path d="M46 330 L430 330" opacity=".65"/>
    <path d="M104 258 L380 258" opacity=".45"/>
    <path d="M152 198 L340 198" opacity=".3"/>
  </g>
  <!-- colonnade -->
  <g fill="#ded6c6" stroke="#cec5b3" stroke-width="1.2">
    <rect x="96" y="150" width="24" height="164" rx="5"/>
    <rect x="356" y="150" width="24" height="164" rx="5"/>
  </g>
  <!-- the landmark it is heading for, at the end of the corridor -->
  <rect x="246" y="140" width="70" height="24" rx="5" fill="#b4551a"/>

  <!-- the route, dotted along the floor, clear of the robot -->
  <path d="M214 336 C 244 296, 268 232, 280 186" stroke="#1c6e8c" stroke-width="7"
    fill="none" stroke-linecap="round" stroke-dasharray="0.1 17"/>

  <!-- the waypoint the model points at -->
  <g>
    <circle cx="281" cy="156" r="26" fill="none" stroke="#f4f1ea" stroke-width="9"/>
    <circle cx="281" cy="156" r="26" fill="none" stroke="#a32d2d" stroke-width="4.5"/>
    <g stroke="#f4f1ea" stroke-width="9" stroke-linecap="round">
      <path d="M242 156 h12"/><path d="M308 156 h12"/>
      <path d="M281 117 v12"/><path d="M281 183 v12"/></g>
    <g stroke="#a32d2d" stroke-width="4.5" stroke-linecap="round">
      <path d="M242 156 h12"/><path d="M308 156 h12"/>
      <path d="M281 117 v12"/><path d="M281 183 v12"/></g>
  </g>

  <!-- the robot, from behind and slightly left, so nothing overlaps -->
  <g transform="translate(-40,0)">
    <ellipse cx="222" cy="380" rx="66" ry="11" fill="#c6bda9" opacity=".5"/>
    <rect x="164" y="330" width="24" height="46" rx="12" fill="#2a303c"/>
    <rect x="256" y="330" width="24" height="46" rx="12" fill="#2a303c"/>
    <rect x="177" y="288" width="90" height="86" rx="20" fill="#b4551a"/>
    <rect x="177" y="288" width="90" height="28" rx="14" fill="#c9662a"/>
    <rect x="213" y="262" width="18" height="34" rx="8" fill="#8f4415"/>
    <rect x="180" y="216" width="84" height="56" rx="16" fill="#2a303c"/>
    <rect x="180" y="216" width="84" height="18" rx="9" fill="#3b4250"/>
    <!-- one RGB camera, which is the entire argument of the paper -->
    <circle cx="222" cy="245" r="18" fill="#11151c"/>
    <circle cx="222" cy="245" r="11.5" fill="#1c6e8c"/>
    <circle cx="222" cy="245" r="5" fill="#0d3a4b"/>
    <circle cx="217" cy="240" r="3.2" fill="#eaf4f8" opacity=".92"/>
  </g>

  <!-- the sight line from that camera to the pixel it chose -->
  <path d="M196 240 L256 162" stroke="#a32d2d" stroke-width="2"
    stroke-dasharray="5 6" fill="none" opacity=".65"/>
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
