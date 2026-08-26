#!/usr/bin/env python3
"""Assemble the Shieldstral visual guide into one self-contained index.html.

  tokens.css      design system
  beats.html      34 numbered beats in 4 acts, text only
  engine.js       scroll engine, scene registry, rail, status
  scenes/*.js     the live art, one file per act
  scenes/*.css    scoped styles for those scenes
  data.js         window.SS, generated from the paper by the two scripts here
"""
import pathlib
import re

HERE = pathlib.Path(__file__).resolve().parent
SCENES = HERE / "scenes"

tokens = (HERE / "tokens.css").read_text(encoding="utf-8")
beats = (HERE / "beats.html").read_text(encoding="utf-8")
engine = (HERE / "engine.js").read_text(encoding="utf-8")
kit = (HERE / "kit.js").read_text(encoding="utf-8")
tiles = (HERE / "tiles.js").read_text(encoding="utf-8")
data = (HERE / "data.js").read_text(encoding="utf-8")

# one file pair per scene, so a single broken scene cannot take the rest down

scene_css, scene_js, acts = [], [], []
for j in sorted(SCENES.glob("*.js")):
    n = j.stem
    c = j.with_suffix(".css")
    if c.exists():
        scene_css.append(f"/* ---- {n} ---- */\n" + c.read_text(encoding="utf-8").strip())
    scene_js.append(f"/* ---- {n} ---- */\n" + j.read_text(encoding="utf-8").strip())
    acts.append(n)

wanted = re.findall(r'data-s="(\w+)"', beats)
have = set()
for j in scene_js:
    have |= set(re.findall(r"SCENES\[['\"](\w+)['\"]\]", j))   # SCENES['ID']
    have |= set(re.findall(r"SCENES\.(\w+)\s*=", j))               # SCENES.ID =
    have |= set(re.findall(r"^\s*(\w+)\s*:\s*function", j, re.M))
missing = [w for w in wanted if w not in have]

TITLE = "Shieldstral, taken apart"
DESC = ("An open-weight moderation model that reads its rule as plain text at inference "
        "time. Thirty-four live figures on the method, the data pipeline, the training, "
        "and where the numbers are less flattering.")

URL = "https://avisoori1x.github.io/shieldstral/"

HERO = """
<div class="hero">
  <div class="eyebrow">a visual guide &nbsp;&middot;&nbsp; paper results, with labelled schematics</div>
  <h1>Shieldstral,<br>taken <em>apart</em></h1>
  <p class="sub">Shieldstral is an open-weight <b>moderation model</b>. It carries no fixed list of
  harm categories. You hand it the moderation criterion as plain text at inference time and it
  answers one yes/no question about one document.</p>
  <p class="sub" style="margin-top:14px">I worked on Shieldstral, and this is my attempt to lay the
  paper out visually. Every figure is live, so you can scroll through the method and try each piece
  as you get to it.</p>

  <div class="metastrip">
    <span class="chip"><a href="https://arxiv.org/abs/2607.25857">arXiv <b>2607.25857</b></a></span>
    <span class="chip"><a href="https://huggingface.co/mistralai/Shieldstral-1.0-3B">weights <b>Apache 2.0</b></a></span>
    <span class="chip">text F1 <b>84.9</b></span>
    <span class="chip">multimodal F1 <b>83.8</b></span>
  </div>
  <div class="toc">
    <div class="a1"><h4>The idea</h4>
      <p>Why a fixed list of harms breaks, moderation as one yes/no question, the three fields, two
      logits, and one document answered yes and no depending on the question.</p></div>
    <div class="a2"><h4>The data</h4>
      <p>Per-dataset processors, strictness tiers, seven document formats, hard negatives, the LLM
      rewrite loop, free positives up the tree, and why the image data had to be built
      differently.</p></div>
    <div class="a3"><h4>The training</h4>
      <p>What the checkpoint actually is, LoRA against full SFT, two checkpoints wrong in
      opposite directions, and the SLERP merge that ships instead of either.</p></div>
    <div class="a4"><h4>Does it hold up</h4>
      <p>An evaluation taxonomy built not to match the training one, the benchmark numbers, the
      adaptability loss to a 20B model, the language holes, and how to run it.</p></div>
  </div>
  <div class="scrollcue"><i></i> scroll &nbsp;&middot;&nbsp; the figure on the right is live</div>
</div>
"""

OUTRO = """
<div class="outro">
  <h2>Notes</h2>
  <p>Numbers and section references come from the paper,
  <a href="https://arxiv.org/abs/2607.25857">arXiv 2607.25857</a>, and the
  <a href="https://huggingface.co/mistralai/Shieldstral-1.0-3B">model card</a>. Where the paper does
  not report something, the beat says so instead of guessing.</p>
  <p>If a figure here disagrees with the paper, the paper is right. Tell me and I will fix it.</p>
  <p>Avinash Sooriyarachchi</p>
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
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Sora:wght@400;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&family=JetBrains+Mono:wght@400;500;700&display=swap">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect width=%22100%22 height=%22100%22 fill=%22%23090b11%22/><text y=%22.92em%22 x=%22.08em%22 font-size=%2288%22 font-family=%22monospace%22 fill=%22%234ec9f5%22>S</text></svg>">
<style>
{tokens}

/* ======================= scenes ======================= */
{chr(10).join(scene_css)}
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

<script>
{data}
</script>
<script>
{tiles}
{kit}
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
print(f"acts wired: {', '.join(acts) or 'none'}")
print(f"scenes: {len(wanted) - len(missing)}/{len(wanted)}")
if missing:
    print("MISSING SCENES:", ", ".join(missing))
