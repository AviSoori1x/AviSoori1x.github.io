#!/usr/bin/env python3
"""Assemble the Robostral Navigate visual guide into one index.html.

  tokens.css   design system, shared with the Shieldstral guide
  beats.html   the numbered beats, text only
  kit.js       the 2D SVG drawing vocabulary
  kit3.js      the three.js layer, one shared WebGL context
  engine.js    scroll engine and scene registry
  scenes/*.js  the figures
  data.js      window.RN, generated from the paper by the two parse scripts

three.js loads as an ES module from a CDN. The UMD build still exists at r160
but prints a deprecation warning to the console on every load, and the
verification harness treats console noise as a defect.
"""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
SCENES = HERE / "scenes"

tokens = (HERE / "tokens.css").read_text(encoding="utf-8")
beats = (HERE / "beats.html").read_text(encoding="utf-8")
engine = (HERE / "engine.js").read_text(encoding="utf-8")
kit = (HERE / "kit.js").read_text(encoding="utf-8")
kit3 = (HERE / "kit3.js").read_text(encoding="utf-8")
data = (HERE / "data.js").read_text(encoding="utf-8")
extra = (HERE / "scene.css")
extra = extra.read_text(encoding="utf-8") if extra.exists() else ""

scene_js, files = [], []
for j in sorted(SCENES.glob("*.js")):
    scene_js.append(f"/* ---- {j.stem} ---- */\n" + j.read_text(encoding="utf-8").strip())
    files.append(j.stem)

wanted = re.findall(r'data-s="(\w+)"', beats)
have = set()
for s in scene_js:
    have |= set(re.findall(r"SCENES\[['\"](\w+)['\"]\]", s))
    have |= set(re.findall(r"SCENES\.(\w+)\s*=", s))
missing = [w for w in wanted if w not in have]
orphan = sorted(have - set(wanted))

TITLE = "Robostral Navigate, taken apart"
DESC = ("An 8B vision-language model that navigates from one RGB camera by pointing at "
        "where to go next. A scrollable walk through the paper, with the 3D scenes live "
        "so you can move the camera and watch the waypoint move with it.")
URL = "https://avisoori1x.github.io/robostral/"
THREE_URL = "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js"

HERO = """
<div class="hero">
  <div class="eyebrow">a visual guide &nbsp;&middot;&nbsp; paper results, live 3D figures</div>
  <h1>Robostral Navigate,<br>taken <em>apart</em></h1>
  <p class="sub">Most navigation systems that work well lean on depth sensors, LiDAR, camera rigs or
  a map built in advance. Every one of those narrows the set of robots you can run on. Robostral
  Navigate takes a <b>single RGB camera</b> and predicts where to go by <b>pointing at a pixel</b>
  in the frame it is already looking at.</p>
  <p class="sub" style="margin-top:14px">I contributed to the original paper, and this is my
  attempt to lay it out visually. The 3D figures are generated scenes rather than diagrams, so where a figure
  is about an image coordinate you can move the camera and watch that coordinate move.</p>

  <div class="metastrip">
    <span class="chip"><a href="https://arxiv.org/abs/2607.20785">arXiv <b>2607.20785</b></a></span>
    <span class="chip"><a href="https://mistral.ai/news/robostral-navigate">project page</a></span>
    <span class="chip">R2R-CE val unseen SR <b>77.4</b></span>
    <span class="chip">RxR-CE val unseen SR <b>75.1</b></span>
  </div>
  <div class="toc">
    <div class="a1"><h4>The problem</h4>
      <p>What every extra sensor costs you, why a metric waypoint does not survive a change of
      robot, and what it means to point at a pixel instead.</p></div>
    <div class="a2"><h4>How it moves</h4>
      <p>A big model thinking slowly on top of a small one acting fast, the five numbers it emits,
      the fallback for when the goal is behind you, and one policy on two very different bodies.</p></div>
    <div class="a3"><h4>How it learns</h4>
      <p>2.4 million simulated trajectories, episode packing that turns quadratic training into
      linear, a mask that stops the model cheating off its own past answers, and the reinforcement
      learning pass that adds what shortest-path imitation never shows.</p></div>
    <div class="a4"><h4>Does it hold up</h4>
      <p>Fifteen baselines, the gap to systems that get depth for free, what reinforcement learning
      actually bought, and the one number where it still comes third.</p></div>
  </div>
  <div class="scrollcue"><i></i> scroll &nbsp;&middot;&nbsp; the figure on the right is live</div>
</div>
"""

OUTRO = """
<div class="outro">
  <h2>Notes</h2>
  <p>Every number in the figures is parsed straight out of the paper,
  <a href="https://arxiv.org/abs/2607.20785">arXiv 2607.20785</a>, rather than typed in by hand.
  The prose around them is my reading of the paper, not the paper's words, so check anything that
  matters against the source.</p>
  <p>The 3D scenes are generated, not recordings. The image coordinates in them come from
  projecting a world point through the camera you are looking through, which is real geometry, but
  the rooms are invented and none of it is a frame from a real robot.</p>
  <p>If a figure here disagrees with the paper, the paper is right. Tell me and I will fix it.</p>
</div>
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{DESC}">
<link rel="canonical" href="{URL}">
<meta property="og:type" content="article">
<meta property="og:site_name" content="Avinash Sooriyarachchi">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{URL}">
<meta property="og:locale" content="en_US">
<meta property="og:image" content="{URL}og.png">
<meta property="og:image:secure_url" content="{URL}og.png">
<meta property="og:image:type" content="image/png">
<meta property="og:image:width" content="1200">
<meta property="og:image:height" content="630">
<meta property="og:image:alt" content="{TITLE}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:site" content="@aviTwit3">
<meta name="twitter:creator" content="@aviTwit3">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{DESC}">
<meta name="twitter:image" content="{URL}og.png">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="preconnect" href="https://cdn.jsdelivr.net" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect width=%22100%22 height=%22100%22 fill=%22%23141922%22/><circle cx=%2250%22 cy=%2250%22 r=%2216%22 fill=%22none%22 stroke=%22%23b4551a%22 stroke-width=%228%22/></svg>">
<style>
{tokens}

/* ======================= 3D figures ======================= */
.k3 {{ background: var(--bg); border-radius: 12px; overflow: hidden; }}
.k3ov text {{ font-family: 'JetBrains Mono', monospace; }}
.scene .ctl {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin-top:14px; }}
.scene .ctl label {{ font: 500 11px/1 'JetBrains Mono',monospace; letter-spacing:.1em;
  text-transform:uppercase; color: var(--ink3); display:flex; align-items:center; gap:8px; }}
.scene .ctl input[type=range] {{ width: 132px; accent-color: var(--blue); }}
.scene .ctl output {{ font: 500 12px/1 'JetBrains Mono',monospace; color: var(--ink);
  min-width: 52px; }}
.scene .seg {{ display:inline-flex; border:1px solid var(--line); border-radius:999px;
  overflow:hidden; }}
.scene .seg button {{ font: 500 11.5px/1 'JetBrains Mono',monospace; padding:7px 13px;
  border:0; background:transparent; color:var(--ink3); cursor:pointer; }}
.scene .seg button[aria-pressed=true] {{ background: var(--blue); color:#fff; }}
{extra}
</style>
</head>
<body>
{HERO}

<div id="stage"><div id="art"></div></div>
<div class="railwrap"><div id="rail"></div></div>
<div id="actlab"></div>

<div id="beats">
{beats}
</div>

{OUTRO}

<script type="module">
import * as THREE from '{THREE_URL}';
window.THREE = THREE;
document.dispatchEvent(new Event('three:ready'));
</script>
<script>
{data}
</script>
<script>
{kit}
{kit3}
window.SCENES = window.SCENES || {{}};
{chr(10).join(scene_js)}
</script>
<script>
{engine}
</script>
</body>
</html>
"""

(HERE / "index.html").write_text(html, encoding="utf-8")
print(f"assembled index.html ({len(html):,} bytes)")
print(f"scene files: {', '.join(files) or 'none'}")
print(f"scenes wired: {len(wanted) - len(missing)}/{len(wanted)}")
if missing:
    print("MISSING SCENES:", ", ".join(missing))
if orphan:
    print("defined but unused:", ", ".join(orphan))
