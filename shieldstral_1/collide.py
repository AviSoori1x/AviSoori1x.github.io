#!/usr/bin/env python3
"""Strict collision gate. Any overlapping text is a failure.

Written after shipping a page with letters on top of letters while an earlier
check reported zero problems. That check required 40 percent area overlap of the
smaller box, which missed a 58px numeral running through a caption. This one
fails on any overlap wider than 2px and taller than 4px, and on any text that
crosses the right edge of its board.

  python3 collide.py [url]
"""
import json
import subprocess
import sys
from pathlib import Path

VENV = "/mnt/vast/home/avi/science/website/.venv/bin/python"
URL = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8412/shieldstral_1/"

PROBE = r'''
import json, sys
from playwright.sync_api import sync_playwright
url = sys.argv[1]
out = {}
with sync_playwright() as p:
    b = p.chromium.launch()
    pg = b.new_context(viewport={"width":1440,"height":900}).new_page()
    errs = []
    pg.on("pageerror", lambda e: errs.append(str(e)[:90]))
    pg.on("console", lambda m: errs.append("console: " + m.text[:90]) if m.type == "error" else None)
    pg.goto(url, wait_until="networkidle")
    pg.add_style_tag(content="html{scroll-behavior:auto !important}")
    pg.wait_for_timeout(900)
    for sid in pg.eval_on_selector_all(".beat", "es=>es.map(e=>e.dataset.s)"):
        pg.evaluate("""(s)=>{const e=document.querySelector('.beat[data-s="'+s+'"]');
          const r=e.getBoundingClientRect();
          window.scrollTo(0,window.scrollY+r.top+r.height/2-window.innerHeight/2);}""", sid)
        pg.wait_for_timeout(430)
        out[sid] = pg.evaluate("""(sid)=>{
          const s=document.querySelector('.scene[data-scene="'+sid+'"]');
          if(!s) return ['scene missing'];
          const sv=s.querySelector('svg');
          if(!sv) return ['no board'];
          const sb=sv.getBoundingClientRect();
          const tx=[...s.querySelectorAll('text')].filter(e=>e.textContent.trim());
          const bx=tx.map(e=>({r:e.getBoundingClientRect(), t:e.textContent.slice(0,30)}));
          const hits=[];
          for(let i=0;i<bx.length;i++){
            if(bx[i].r.right > sb.right+1) hits.push('off the right edge: "'+bx[i].t+'"');
            if(bx[i].r.left < sb.left-1)  hits.push('off the left edge: "'+bx[i].t+'"');
            for(let j=i+1;j<bx.length;j++){
              const a=bx[i].r,c=bx[j].r;
              const ow=Math.min(a.right,c.right)-Math.max(a.left,c.left);
              const oh=Math.min(a.bottom,c.bottom)-Math.max(a.top,c.top);
              if(ow>2 && oh>4) hits.push('"'+bx[i].t+'" over "'+bx[j].t+'"');
            }
          }
          return hits;}""", sid)
    out["__errors__"] = errs
    b.close()
print(json.dumps(out))
'''


def main():
    probe = Path("/tmp/.collide_probe.py")
    probe.write_text(PROBE)
    res = subprocess.run([VENV, str(probe), URL], capture_output=True, text=True, timeout=900)
    probe.unlink(missing_ok=True)
    if res.returncode != 0:
        print(res.stderr[-1200:])
        sys.exit(2)
    data = json.loads(res.stdout.strip().splitlines()[-1])
    errs = data.pop("__errors__", [])

    total = 0
    for sid, hits in data.items():
        if not hits:
            continue
        total += len(hits)
        print(f"{sid}  ({len(hits)})")
        for h in hits[:4]:
            print("    " + h[:110])

    if errs:
        print("\njs errors:")
        for e in errs[:5]:
            print("    " + e)

    print()
    if total or errs:
        print(f"FAIL: {total} text collisions across {len([1 for v in data.values() if v])} figures")
        sys.exit(1)
    print(f"clean: {len(data)} figures, no overlapping text, no js errors")


if __name__ == "__main__":
    main()
