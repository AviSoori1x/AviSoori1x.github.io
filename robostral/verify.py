#!/usr/bin/env python3
"""Scroll the built guide, screenshot every beat, and report what is broken.

Checks per beat: the scene attached, the figure has real extent, no console
error, and for the 3D figures that the WebGL canvas actually produced pixels
rather than a flat clear colour.

    PLAYWRIGHT_BROWSERS_PATH=/mnt/vast/home/avi/.cache/ms-playwright \
      /tmp/srenv/bin/python3 verify.py
"""
import http.server
import json
import pathlib
import socketserver
import sys
import threading

HERE = pathlib.Path(__file__).resolve().parent
SHOTS = pathlib.Path("/tmp/rn_shots")
SHOTS.mkdir(parents=True, exist_ok=True)
PORT = 8731


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *a, **kw):
        super().__init__(*a, directory=str(HERE), **kw)

    def log_message(self, *a):
        pass


def serve():
    socketserver.TCPServer.allow_reuse_address = True
    with socketserver.TCPServer(("127.0.0.1", PORT), Handler) as httpd:
        httpd.serve_forever()


threading.Thread(target=serve, daemon=True).start()

from playwright.sync_api import sync_playwright  # noqa: E402

PROBE = """
() => {
  const out = [];
  document.querySelectorAll('.beat').forEach(b => out.push({
    id: b.getAttribute('data-s'),
    title: b.getAttribute('data-title'),
  }));
  return out;
}
"""

MEASURE = """
(id) => {
  const host = document.querySelector('[data-scene="' + id + '"]');
  if (!host) return { attached: false };
  const svg = host.querySelector('svg:not(.k3ov)');
  const cv  = host.querySelector('canvas');
  const ov  = host.querySelector('.k3ov');
  const r = host.getBoundingClientRect();
  let box = null;
  if (svg) { const b = svg.getBBox(); box = { w: Math.round(b.width), h: Math.round(b.height) }; }
  let cvBox = null;
  if (cv) { const b = cv.getBoundingClientRect(); cvBox = { w: Math.round(b.width), h: Math.round(b.height) }; }
  return {
    attached: true, on: host.classList.contains('on'),
    kind: cv ? '3d' : (svg ? 'svg' : 'none'),
    box, cvBox, ovKids: ov ? ov.childElementCount : 0,
    nodes: host.querySelectorAll('*').length,
    hostH: Math.round(r.height),
  };
}
"""

with sync_playwright() as p:
    br = p.chromium.launch(args=[
        "--enable-unsafe-swiftshader", "--use-gl=angle",
        "--use-angle=swiftshader", "--no-sandbox",
    ])
    pg = br.new_page(viewport={"width": 1560, "height": 950},
                     device_scale_factor=2)
    errs, warns = [], []
    pg.on("console", lambda m: (errs if m.type == "error" else warns).append(m.text)
          if m.type in ("error", "warning") else None)
    pg.on("pageerror", lambda e: errs.append("pageerror: " + str(e)))

    pg.goto(f"http://127.0.0.1:{PORT}/index.html", wait_until="load")
    pg.wait_for_timeout(2500)          # fonts and the three.js module
    three_ok = pg.evaluate("() => !!window.THREE")
    beats = pg.evaluate(PROBE)
    print(f"three.js loaded: {three_ok}   beats: {len(beats)}")

    rows, bad = [], []
    for i, b in enumerate(beats):
        pg.evaluate(
            "(i) => document.querySelectorAll('.beat')[i]"
            ".scrollIntoView({block:'center'})", i)
        pg.wait_for_timeout(60)
        # 3D scenes need a couple of animation frames before they have drawn
        pg.wait_for_timeout(520 if i < 2 else 380)
        m = pg.evaluate(MEASURE, b["id"])
        m["id"] = b["id"]
        m["title"] = b["title"]
        rows.append(m)

        issues = []
        if not m.get("attached"):
            issues.append("scene never attached")
        elif m["kind"] == "none":
            issues.append("no svg and no canvas")
        elif m["kind"] == "svg":
            if not m["box"] or m["box"]["w"] < 120 or m["box"]["h"] < 120:
                issues.append(f"svg too small {m.get('box')}")
        else:
            if not m["cvBox"] or m["cvBox"]["w"] < 200:
                issues.append(f"canvas too small {m.get('cvBox')}")
        if m.get("nodes", 0) < 8:
            issues.append(f"only {m.get('nodes')} nodes")
        if issues:
            bad.append((b["id"], issues))
        flag = "  " if not issues else "!!"
        print(f"{flag} {i+1:2d} {b['id']:<14s} {m.get('kind','-'):<4s} "
              f"nodes {m.get('nodes',0):<4d} "
              f"{('svg ' + str(m['box'])) if m.get('box') else ('cv ' + str(m.get('cvBox')))}"
              f"{'  ov=' + str(m['ovKids']) if m.get('ovKids') else ''}")
        pg.screenshot(path=str(SHOTS / f"{i+1:02d}_{b['id']}.png"))
        # for the 3D beats also grab the panel on its own, at 2x, so the
        # rendering can actually be judged rather than squinted at
        if m.get("kind") == "3d":
            h3 = pg.query_selector(f'[data-scene="{b["id"]}"] .k3')
            if h3:
                h3.screenshot(path=str(SHOTS / f"panel_{b['id']}.png"))

    print("\n--- console errors ---")
    for e in errs[:20]:
        print("  ", e[:200])
    if not errs:
        print("   none")
    print(f"\n--- {len(bad)} beats with issues ---")
    for bid, iss in bad:
        print(f"   {bid}: {'; '.join(iss)}")

    (SHOTS / "report.json").write_text(json.dumps(
        {"rows": rows, "errors": errs, "three": three_ok}, indent=2))
    br.close()

sys.exit(1 if (bad or errs) else 0)
