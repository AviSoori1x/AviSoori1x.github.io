#!/usr/bin/env python3
"""Static site generator for avisoori1x.github.io.

Sources live in _src/ (templates, css, yaml data) and _posts/ (markdown).
Output is written to the repo root so GitHub Pages can serve it directly;
.nojekyll stops Pages from trying to run Jekyll over it.

Post URLs reproduce the historical Jekyll permalinks exactly
(/YYYY/MM/DD/<name>.html) so no inbound link breaks.

    python build.py            # build
    python build.py --serve    # build, then serve on :8000
"""

from __future__ import annotations

import argparse
import datetime as dt
import html
import re
import shutil
import sys
from pathlib import Path

import markdown
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).parent.resolve()
SRC = ROOT / "_src"
POSTS = ROOT / "_posts"
OUT = ROOT

# Directories/files the build owns and may overwrite.
GENERATED = ["assets", "feed.xml", "sitemap.xml", "index.html", "404.html", "robots.txt"]


# --------------------------------------------------------------------------- data


def load_data() -> dict:
    data = {}
    for f in sorted((SRC / "data").glob("*.yml")):
        with f.open(encoding="utf-8") as fh:
            data[f.stem] = yaml.safe_load(fh)
    return data


# --------------------------------------------------------------------------- posts

POST_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})-(.+)$")


def legacy_url(stem: str) -> str:
    """Reproduce the Jekyll permalink for a post filename stem.

    Verified against the live feed: ':' and whitespace become '-', every other
    character (including ',' and '_') and the original casing are preserved.
    """
    m = POST_RE.match(stem)
    if not m:
        raise ValueError(f"post filename is not date-prefixed: {stem}")
    y, mo, d, name = m.groups()
    return f"/{y}/{mo}/{d}/{re.sub(r'[:\s]', '-', name)}.html"


def read_posts(titles: dict[str, str]) -> list[dict]:
    md = markdown.Markdown(
        extensions=["fenced_code", "codehilite", "tables", "attr_list", "md_in_html", "sane_lists"],
        extension_configs={"codehilite": {"css_class": "highlight", "guess_lang": False}},
    )
    posts = []
    for f in sorted(POSTS.glob("*.md")):
        stem = f.stem
        m = POST_RE.match(stem)
        if not m:
            continue
        y, mo, d, _ = m.groups()
        raw = f.read_text(encoding="utf-8")

        # Jekyll's optional-front-matter plugin allowed these files to omit it.
        raw = re.sub(r"\A---\n.*?\n---\n", "", raw, flags=re.S)

        # jekyll-titles-from-headings pulled the title out of the leading H1
        # and stripped it from the body; reproduce that.
        title = titles.get(stem)
        h1 = re.match(r"\s*#\s+(.+?)\s*\n", raw)
        if h1:
            if not title:
                title = h1.group(1).strip()
            raw = raw[h1.end():]
        if not title:
            raise SystemExit(f"no title for post {stem} — add one to data/posts.yml")

        md.reset()
        body = md.convert(raw)
        words = len(re.sub(r"<[^>]+>", " ", body).split())
        date = dt.date(int(y), int(mo), int(d))
        posts.append(
            {
                "stem": stem,
                "title": title,
                "date": date,
                "date_display": date.strftime("%d %B %Y").lstrip("0"),
                "year": date.year,
                "url": legacy_url(stem),
                "html": body,
                "reading": max(1, round(words / 220)),
                "has_math": "$$" in raw or r"\(" in raw,
                "excerpt": excerpt_of(body),
                "blurb": excerpt_of(body, 155),
            }
        )
    posts.sort(key=lambda p: p["date"], reverse=True)
    return posts


def excerpt_of(body_html: str, limit: int = 240) -> str:
    text = re.sub(r"<(script|style|iframe)[^>]*>.*?</\1>", " ", body_html, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    text = re.sub(r"^TL;DR:\s*", "", text)
    return (text[:limit].rsplit(" ", 1)[0] + "…") if len(text) > limit else text


# --------------------------------------------------------------------------- render


def build(serve: bool = False) -> None:
    data = load_data()
    site = data["site"]
    site["year"] = dt.date.today().year

    env = Environment(
        loader=FileSystemLoader(SRC / "templates"),
        autoescape=select_autoescape(["html"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    # Root-relative internal links break under a subpath deploy (preview builds),
    # so emit them relative to the current page instead.
    env.filters["u"] = lambda s, rel="": (rel + s.lstrip("/")) if s.startswith("/") else s

    posts = read_posts(data.get("posts", {}).get("titles", {}) or {})

    # ---- assets
    assets = OUT / "assets"
    assets.mkdir(exist_ok=True)
    css = (SRC / "css" / "main.css").read_text(encoding="utf-8")
    css += "\n\n/* ---- pygments ---- */\n" + pygments_css()
    (assets / "main.css").write_text(css, encoding="utf-8")

    # ---- index: posts on this site are merged into the writing list
    writing = list(data["writing"]["items"])
    folded = set(data.get("posts", {}).get("secondary") or [])
    for p in posts:
        writing.append(
            {
                "title": p["title"],
                "url": p["url"],
                "date": p["date"].strftime("%Y"),
                "sort": p["date"].isoformat(),
                "publisher": "This site",
                "note": p["blurb"],
                "secondary": p["stem"] in folded,
            }
        )
    writing.sort(key=lambda w: str(w.get("sort") or w.get("date")), reverse=True)
    # "secondary" pieces stay in the fold so the selected list reads as research work
    writing_primary = [w for w in writing if not w.get("secondary")]
    writing_more = [w for w in writing if w.get("secondary")]

    talks = sorted(data["talks"]["items"], key=lambda t: str(t.get("sort") or t.get("date")), reverse=True)
    media = sorted(data["media"]["items"], key=lambda m: str(m.get("sort") or m.get("date")), reverse=True)

    index_html = env.get_template("index.html").render(
        site=site,
        profile=data["profile"]["profile"],
        now=data["profile"]["now"],
        projects=data["projects"]["items"],
        research=sorted(
            data["research"]["items"], key=lambda r: str(r.get("sort") or r.get("date")), reverse=True
        ),
        research_note=data["research"].get("note"),
        writing_primary=writing_primary,
        writing_more=writing_more,
        talks=talks,
        media=media,
        timeline=data["timeline"]["items"],
        page_title=f"{site['name']} · {data['profile']['profile']['role']}, {data['profile']['profile']['org']}",
        page_desc=data["profile"]["profile"]["meta_desc"],
        canonical="/",
        rel="",
        json_ld=person_json_ld(site, data["profile"]["profile"]),
    )
    (OUT / "index.html").write_text(index_html, encoding="utf-8")

    # ---- posts
    tpl = env.get_template("post.html")
    for p in posts:
        dest = OUT / p["url"].lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        depth = p["url"].strip("/").count("/")
        dest.write_text(
            tpl.render(
                site=site,
                post=p,
                page_title=f"{p['title']} · {site['name']}",
                page_desc=p["excerpt"][:180],
                canonical=p["url"],
                rel="../" * depth,
                og_type="article",
            ),
            encoding="utf-8",
        )

    write_feed(site, posts)
    write_sitemap(site, posts)
    write_404(env, site)
    (OUT / ".nojekyll").write_text("", encoding="utf-8")
    (OUT / "robots.txt").write_text(
        f"User-agent: *\nAllow: /\nSitemap: {site['url']}/sitemap.xml\n", encoding="utf-8"
    )

    print(f"built  index + {len(posts)} posts  ->  {OUT}")
    for p in posts:
        print(f"       {p['url']}")

    if serve:
        import http.server
        import socketserver

        handler = lambda *a, **kw: http.server.SimpleHTTPRequestHandler(*a, directory=str(OUT), **kw)  # noqa: E731
        with socketserver.TCPServer(("127.0.0.1", 8000), handler) as httpd:
            print("serving http://127.0.0.1:8000  (ctrl-c to stop)")
            httpd.serve_forever()


def pygments_css() -> str:
    return HtmlFormatter(style="tango").get_style_defs(".highlight")


def person_json_ld(site: dict, profile: dict) -> str:
    import json

    blob = json.dumps(
        {
            "@context": "https://schema.org",
            "@type": "Person",
            "name": site["name"],
            "url": site["url"],
            "jobTitle": profile["role"],
            "worksFor": {"@type": "Organization", "name": profile["org"]},
            "address": {"@type": "PostalAddress", "addressLocality": profile["location"]},
            "sameAs": [link["url"] for link in site["links"]],
            "description": profile["meta_desc"],
        },
        separators=(",", ":"),
    )
    # emitted unescaped inside a <script>, so neutralise any closing tag
    return blob.replace("</", "<\\/")


def write_feed(site: dict, posts: list[dict]) -> None:
    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    items = []
    for p in posts:
        stamp = dt.datetime.combine(p["date"], dt.time()).isoformat() + "+00:00"
        items.append(
            f"""  <entry>
    <title>{esc(p['title'])}</title>
    <link href="{site['url']}{p['url']}"/>
    <id>{site['url']}{p['url']}</id>
    <published>{stamp}</published>
    <updated>{stamp}</updated>
    <author><name>{esc(site['name'])}</name></author>
    <summary>{esc(p['excerpt'])}</summary>
  </entry>"""
        )
    updated = (
        dt.datetime.combine(posts[0]["date"], dt.time()).isoformat() + "+00:00" if posts else ""
    )
    (OUT / "feed.xml").write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title>{esc(site['name'])}</title>
  <link href="{site['url']}/feed.xml" rel="self"/>
  <link href="{site['url']}/"/>
  <id>{site['url']}/</id>
  <updated>{updated}</updated>
  <author><name>{esc(site['name'])}</name></author>
{chr(10).join(items)}
</feed>
""",
        encoding="utf-8",
    )


def write_sitemap(site: dict, posts: list[dict]) -> None:
    urls = [f"  <url><loc>{site['url']}/</loc></url>"]
    for p in posts:
        urls.append(
            f"  <url><loc>{site['url']}{p['url']}</loc>"
            f"<lastmod>{p['date'].isoformat()}</lastmod></url>"
        )
    (OUT / "sitemap.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        + "\n".join(urls)
        + "\n</urlset>\n",
        encoding="utf-8",
    )


def write_404(env: Environment, site: dict) -> None:
    (OUT / "404.html").write_text(
        env.from_string(
            """{% extends "base.html" %}
{% block content %}
<header class="post-head rise d1">
  <h1>404</h1>
  <div class="post-meta"><span>Page not found</span></div>
</header>
<div class="prose rise d2"><p>That page doesn't exist. Try the <a href="/">index</a>.</p></div>
{% endblock %}"""
        ).render(
            site=site,
            page_title=f"404 · {site['name']}",
            page_desc="Page not found",
            canonical="/404.html",
            rel="",
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve", action="store_true")
    args = ap.parse_args()
    build(serve=args.serve)
