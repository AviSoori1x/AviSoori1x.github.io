#!/usr/bin/env python3
"""Render the 1200x630 social preview card.

LinkedIn, X and Slack all want an og:image at an absolute URL. Rather than
hand-draw one, this builds a card in the page's own design system, screenshots
it with the real fonts, and writes og.png next to the guide.

Run after any change to the title or the palette:
    python3 make_og.py
"""
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
VENV = "/mnt/vast/home/avi/science/website/.venv/bin/python"

CARD = """<!DOCTYPE html><html><head><meta charset="utf-8">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Sans:wght@400;500&family=JetBrains+Mono:wght@400;700&display=swap">
<style>
  *{box-sizing:border-box;margin:0}
  body{width:1200px;height:630px;background:#f4f1ea;font-family:'IBM Plex Sans',sans-serif;
       display:flex;overflow:hidden}
  .left{flex:1;padding:64px 0 56px 68px;display:flex;flex-direction:column;justify-content:center}
  .eyebrow{font-family:'JetBrains Mono',monospace;font-size:17px;letter-spacing:.17em;
           text-transform:uppercase;color:#b4551a;font-weight:700;margin-bottom:24px}
  h1{font-family:'Sora',sans-serif;font-weight:700;font-size:84px;line-height:.98;
     letter-spacing:-.032em;color:#141922;margin-bottom:26px}
  h1 em{font-style:normal;color:#b4551a}
  p{font-size:25px;line-height:1.42;color:#39404f;max-width:19em}
  .chips{display:flex;gap:11px;margin-top:34px}
  .chip{font-family:'JetBrains Mono',monospace;font-size:16px;color:#39404f;background:#fff;
        border:1px solid rgba(31,37,48,.16);border-radius:999px;padding:9px 17px}
  .chip b{color:#1c6e8c}
  .right{width:404px;position:relative;display:flex;align-items:center;justify-content:center}
  .art{width:300px;height:300px;border-radius:14px;overflow:hidden;
       box-shadow:0 26px 60px rgba(20,25,34,.20);border:1px solid rgba(31,37,48,.14);
       transform:rotate(-3deg)}
  .art svg{width:100%;height:100%;display:block}
  .verdict{position:absolute;right:44px;bottom:150px;background:#fff;border:1px solid rgba(31,37,48,.14);
           border-radius:12px;padding:14px 20px;box-shadow:0 16px 40px rgba(20,25,34,.16)}
  .verdict .w{font-family:'Sora',sans-serif;font-weight:700;font-size:44px;line-height:1;color:#0f6e56}
  .verdict .k{font-family:'JetBrains Mono',monospace;font-size:12px;letter-spacing:.16em;
              text-transform:uppercase;color:#0f6e56;margin-top:5px}
  .rail{position:absolute;left:0;top:0;bottom:0;width:8px;background:#b4551a}
</style></head><body>
<div class="rail"></div>
<div class="left">
  <div class="eyebrow">a visual guide</div>
  <h1>Shieldstral,<br>taken <em>apart</em></h1>
  <p>An open-weight moderation model that reads its rule as plain text, in 34 live figures.</p>
  <div class="chips">
    <span class="chip">arXiv <b>2607.25857</b></span>
    <span class="chip">weights <b>Apache 2.0</b></span>
    <span class="chip">text F1 <b>84.9</b></span>
  </div>
</div>
<div class="right">
  <div class="art"><svg viewBox="0 0 300 300" width="300" height="300">
  <defs>
    <linearGradient id="fur" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffd23f"/>
      <stop offset=".45" stop-color="#ff9c1a"/>
      <stop offset="1" stop-color="#e8590c"/>
    </linearGradient>
    <linearGradient id="ear" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ff7a1a"/><stop offset="1" stop-color="#d94a08"/>
    </linearGradient>
  </defs>
  <rect width="300" height="300" fill="#fdf6ec"/>
  <!-- the Mistral banding, as a backdrop -->
  <g opacity=".22">
    <rect x="24" y="228" width="252" height="14" rx="3" fill="#ffd23f"/>
    <rect x="24" y="248" width="252" height="14" rx="3" fill="#ff9c1a"/>
    <rect x="24" y="268" width="252" height="14" rx="3" fill="#e8590c"/>
  </g>
  <!-- ears -->
  <path d="M74 96 L84 34 L136 74 Z" fill="url(#ear)"/>
  <path d="M226 96 L216 34 L164 74 Z" fill="url(#ear)"/>
  <path d="M88 92 L94 56 L124 80 Z" fill="#ffb9a3"/>
  <path d="M212 92 L206 56 L176 80 Z" fill="#ffb9a3"/>
  <!-- head -->
  <ellipse cx="150" cy="150" rx="86" ry="78" fill="url(#fur)"/>
  <!-- stripes -->
  <g stroke="#d94a08" stroke-width="7" stroke-linecap="round" opacity=".55" fill="none">
    <path d="M118 92 L112 74"/><path d="M150 86 L150 66"/><path d="M182 92 L188 74"/>
  </g>
  <!-- eyes -->
  <ellipse cx="120" cy="146" rx="15" ry="19" fill="#1f2430"/>
  <ellipse cx="180" cy="146" rx="15" ry="19" fill="#1f2430"/>
  <circle cx="125" cy="139" r="5" fill="#fff"/>
  <circle cx="185" cy="139" r="5" fill="#fff"/>
  <!-- muzzle -->
  <path d="M150 172 l-11 -9 h22 Z" fill="#c2410c"/>
  <path d="M150 172 v11 M150 183 q-13 10 -24 1 M150 183 q13 10 24 1"
        stroke="#a3350a" stroke-width="5" fill="none" stroke-linecap="round"/>
  <!-- whiskers -->
  <g stroke="#a3350a" stroke-width="4" stroke-linecap="round" opacity=".75">
    <path d="M96 168 L52 158"/><path d="M96 180 L54 182"/>
    <path d="M204 168 L248 158"/><path d="M204 180 L246 182"/>
  </g>
</svg></div>
  <div class="verdict"><div class="w">no</div><div class="k">not flagged</div></div>
</div>
</body></html>"""

SHOT = r'''
import sys, pathlib
from playwright.sync_api import sync_playwright
html, out = sys.argv[1], sys.argv[2]
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context(viewport={"width": 1200, "height": 630}, device_scale_factor=1).new_page()
    pg.goto("file://" + html, wait_until="networkidle")
    pg.wait_for_timeout(1200)
    pg.screenshot(path=out)
    b.close()
print("rendered", out)
'''


def main():
    card = CARD
    tmp = pathlib.Path('/tmp/_og_card.html')
    tmp.write_text(card, encoding='utf-8')
    shot = pathlib.Path('/tmp/_og_shot.py')
    shot.write_text(SHOT)
    out = HERE / 'og.png'
    r = subprocess.run([VENV, str(shot), str(tmp), str(out)],
                       capture_output=True, text=True, timeout=180)
    print(r.stdout.strip() or r.stderr[-600:])
    if out.exists():
        print(f"og.png {out.stat().st_size/1024:.0f} KB")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
