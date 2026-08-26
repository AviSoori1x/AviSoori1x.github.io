#!/usr/bin/env python3
"""Fail the build when a figure is too crowded to read.

Derived from measurement, not taste. Every scene that renders cleanly has fewer
than about 35 text nodes; every scene with overlapping or clipped text has 60 or
more. So the budget is a node count, plus hard zeros for the three defects that
actually make a figure unreadable.

  python3 budget.py            check every scene
  python3 budget.py S_LANG     check one
"""
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
VENV = "/mnt/vast/home/avi/science/website/.venv/bin/python"
URL = "http://127.0.0.1:8412/shieldstral/"

MAX_NODES = 34      # text-bearing leaf elements in the figure
MAX_OVERLAP = 0     # pairs of text boxes sitting on top of each other
MAX_CLIPPED = 0     # elements whose text is cut off by their own box
MAX_OVERFLOW = 8    # px the figure exceeds the stage by
MIN_FONT = 12.5     # px, smallest rendered text in the figure

PROBE = r'''
import json, sys
from playwright.sync_api import sync_playwright
only = sys.argv[1] if len(sys.argv) > 1 else None
out = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context(viewport={"width":1440,"height":900}).new_page()
    pg.goto("URLHERE", wait_until="networkidle")
    pg.add_style_tag(content="html{scroll-behavior:auto !important}")
    pg.wait_for_timeout(800)
    for sid in pg.eval_on_selector_all(".beat","es=>es.map(e=>e.dataset.s)"):
        if only and sid != only: continue
        pg.evaluate("""(s)=>{const e=document.querySelector('.beat[data-s="'+s+'"]');
          const r=e.getBoundingClientRect();
          window.scrollTo(0,window.scrollY+r.top+r.height/2-window.innerHeight/2);}""", sid)
        pg.wait_for_timeout(500)
        out[sid] = pg.evaluate("""(sid)=>{
          const s=document.querySelector('.scene[data-scene="'+sid+'"]');
          if(!s) return {missing:true};
          const leaves=[...s.querySelectorAll('*')].filter(e=>e.children.length===0&&e.textContent.trim());
          let clipped=0, minf=999; const boxes=[];
          for(const e of leaves){
            if(e.scrollWidth>e.clientWidth+2||e.scrollHeight>e.clientHeight+2) clipped++;
            const f=parseFloat(getComputedStyle(e).fontSize); if(f<minf) minf=f;
            const r=e.getBoundingClientRect();
            if(r.width>0&&r.height>0) boxes.push({r,e});
          }
          let overlaps=0;
          for(let i=0;i<boxes.length;i++)for(let j=i+1;j<boxes.length;j++){
            const a=boxes[i],c=boxes[j];
            if(a.e.contains(c.e)||c.e.contains(a.e)) continue;
            const o=Math.max(0,Math.min(a.r.right,c.r.right)-Math.max(a.r.left,c.r.left))
                   *Math.max(0,Math.min(a.r.bottom,c.r.bottom)-Math.max(a.r.top,c.r.top));
            if(o>0.35*Math.min(a.r.width*a.r.height,c.r.width*c.r.height)) overlaps++;
          }
          return {nodes:leaves.length, clipped, overlaps,
                  overflow:Math.max(0,Math.round(s.scrollHeight-s.clientHeight)),
                  minFont:Math.round(minf*10)/10};
        }""", sid)
    b.close()
print(json.dumps(out))
'''.replace("URLHERE", URL)


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    probe = HERE / ".budget_probe.py"
    probe.write_text(PROBE)
    cmd = [VENV, str(probe)] + ([only] if only else [])
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    probe.unlink(missing_ok=True)
    if res.returncode != 0:
        print(res.stderr[-1500:])
        sys.exit(2)
    data = json.loads(res.stdout.strip().splitlines()[-1])

    fails = []
    print(f"{'scene':16s} {'nodes':>6} {'clip':>5} {'ovl':>4} {'ovf':>5} {'minpx':>6}  verdict")
    for sid, m in sorted(data.items(), key=lambda kv: -kv[1].get("nodes", 0)):
        if m.get("missing"):
            fails.append((sid, "scene never rendered"))
            print(f"{sid:16s} {'-':>6} {'-':>5} {'-':>4} {'-':>5} {'-':>6}  MISSING")
            continue
        why = []
        if m["nodes"] > MAX_NODES:      why.append(f"{m['nodes']} text nodes over {MAX_NODES}")
        if m["overlaps"] > MAX_OVERLAP: why.append(f"{m['overlaps']} overlapping")
        if m["clipped"] > MAX_CLIPPED:  why.append(f"{m['clipped']} clipped")
        if m["overflow"] > MAX_OVERFLOW: why.append(f"overflows by {m['overflow']}px")
        if m["minFont"] < MIN_FONT:     why.append(f"{m['minFont']}px type")
        ok = not why
        if not ok:
            fails.append((sid, ", ".join(why)))
        print(f"{sid:16s} {m['nodes']:6d} {m['clipped']:5d} {m['overlaps']:4d} "
              f"{m['overflow']:5d} {m['minFont']:6.1f}  {'ok' if ok else 'FAIL'}")

    print()
    if fails:
        print(f"{len(fails)} of {len(data)} scenes over budget:")
        for sid, why in fails:
            print(f"  {sid:16s} {why}")
        sys.exit(1)
    print(f"all {len(data)} scenes within budget")


if __name__ == "__main__":
    main()
