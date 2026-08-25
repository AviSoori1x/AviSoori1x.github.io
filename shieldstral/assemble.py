#!/usr/bin/env python3
"""Assemble the Shieldstral explainer into one self-contained index.html.

Sources:
  page.css        host design system
  article.html    prose with <!--WIDGET:id--> markers
  page.js         masthead / contents / progress / reveal
  data.js         generated from the paper, defines window.SS
  figures/<id>.{html,css,js}   one interactive figure each
"""
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
WIDGETS = HERE / "figures"

article = (HERE / "article.html").read_text(encoding="utf-8")
page_css = (HERE / "page.css").read_text(encoding="utf-8")
page_js = (HERE / "page.js").read_text(encoding="utf-8")
data_js = (HERE / "data.js").read_text(encoding="utf-8")

# figures whose content is too dense for the 756px text column
WIDE = {
    "w-taxonomy", "w-multilingual", "w-bench", "w-divergence",
    "w-softmax", "w-merge", "w-contrastive", "w-deploy", "w-image",
    "w-fixed-vs-adaptive", "w-policy", "w-anatomy",
}

wanted = re.findall(r"<!--WIDGET:([\w-]+)-->", article)
missing, css_parts, js_parts = [], [], []

for wid in wanted:
    html_p, css_p, js_p = (WIDGETS / f"{wid}.{ext}" for ext in ("html", "css", "js"))
    # a widget is only wired in when all three parts exist, otherwise we would
    # inline dead markup with no behaviour and the figure renders as empty chrome
    if not (html_p.exists() and js_p.exists()):
        missing.append(wid)
        placeholder = (
            f'<figure class="fig" id="{wid}"><figcaption class="fig-head">'
            f'<span class="fig-kicker">PENDING</span>'
            f'<span class="fig-title">{wid}</span></figcaption>'
            f'<p class="fig-note">Figure not yet built.</p></figure>'
        )
        article = article.replace(f"<!--WIDGET:{wid}-->", placeholder)
        continue
    markup = html_p.read_text(encoding="utf-8").strip()
    if wid in WIDE:
        markup = markup.replace('class="fig"', 'class="fig wide"', 1)
    article = article.replace(f"<!--WIDGET:{wid}-->", markup)
    if css_p.exists():
        css_parts.append(f"/* ---- {wid} ---- */\n" + css_p.read_text(encoding="utf-8").strip())
    if js_p.exists():
        js_parts.append(f"/* ---- {wid} ---- */\n" + js_p.read_text(encoding="utf-8").strip())

TITLE = "Shieldstral, explained: how a 3B safety model takes its policy at inference time"
DESC = ("An interactive walkthrough of Shieldstral, Mistral AI's 3B policy-adaptive multimodal "
        "safety classifier: the binary question-answering reformulation, the 54.1M-sample data "
        "curation recipe, the training and merge, and what the benchmarks do and do not show. "
        "By Avinash Sooriyarachchi, a core contributor to the model.")
URL = "https://avisoori1x.github.io/shieldstral/"
IMG = "https://avisoori1x.github.io/images/blogImage.png"

LD = (
    '{"@context":"https://schema.org","@type":"TechArticle",'
    f'"headline":{TITLE!r},'.replace("'", '"') +
    f'"description":{DESC!r},'.replace("'", '"') +
    '"author":{"@type":"Person","name":"Avinash Sooriyarachchi",'
    '"url":"https://avisoori1x.github.io/"},'
    '"datePublished":"2026-08-25","inLanguage":"en",'
    f'"mainEntityOfPage":{{"@type":"WebPage","@id":"{URL}"}},'
    '"about":{"@type":"SoftwareApplication","name":"Shieldstral 1.0 3B",'
    '"url":"https://huggingface.co/mistralai/Shieldstral-1.0-3B"},'
    '"citation":"https://arxiv.org/abs/2607.25857"}'
)

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<meta name="description" content="{DESC}">
<link rel="canonical" href="{URL}">
<meta name="author" content="Avinash Sooriyarachchi">

<meta property="og:type" content="article">
<meta property="og:title" content="{TITLE}">
<meta property="og:description" content="{DESC}">
<meta property="og:url" content="{URL}">
<meta property="og:site_name" content="Avinash Sooriyarachchi">
<meta property="og:image" content="{IMG}">
<meta name="twitter:card" content="summary_large_image">
<meta name="twitter:creator" content="@aviTwit3">
<meta name="twitter:title" content="{TITLE}">
<meta name="twitter:description" content="{DESC}">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Instrument+Serif:ital@0;1&amp;family=Newsreader:ital,opsz,wght@0,6..72,300..700;1,6..72,300..600&amp;family=IBM+Plex+Mono:wght@400;500;600&amp;display=swap">
<link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><rect width=%22100%22 height=%22100%22 fill=%22%23faf7f1%22/><text y=%22.9em%22 x=%22.06em%22 font-size=%2286%22 font-family=%22Georgia,serif%22 fill=%22%23c3341a%22>S</text></svg>">
<script type="application/ld+json">{LD}</script>
<style>
{page_css}

/* ======================= figures ======================= */
{chr(10).join(css_parts)}
</style>
</head>
<body>
<div id="progress" role="presentation"></div>
<div class="page">

<div class="topbar">
  <a class="home" href="/">&#8592; Avinash Sooriyarachchi</a>
  <span class="stamp">Interactive walkthrough</span>
</div>

{article}

</div>
<script>
{data_js}
</script>
<script>
{page_js}

/* ======================= figures ======================= */
{chr(10).join(js_parts)}
</script>
</body>
</html>
"""

(HERE / "index.html").write_text(html, encoding="utf-8")

print(f"assembled index.html  ({len(html):,} bytes)")
print(f"figures wired: {len(wanted) - len(missing)}/{len(wanted)}")
if missing:
    print("MISSING:", ", ".join(missing))
    sys.exit(0)
