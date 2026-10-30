"""A page of figures in cora.earth's style: both themes of every figure inline, the one matching the reader's
setting shown, with the site's light/dark toggle and hover text on marked points.

    html.page("Gaia DR4 candidates", sections, "out/index.html", intro="...")

sections: [{"heading": str, "text": str (HTML), "figures": [sheet entries]}]. Figures are inlined as SVG (their
ids prefixed per figure so two copies never clash); `png_links=True` adds links to the PNGs beside the page.
"""

import html as _html
import re
from pathlib import Path

from cosmos_search.viz import style

_ID = re.compile(r'\bid="([^"]+)"')
_NUM = re.compile(r"(\d+\.\d{2})\d+")


def _tokens_css():
    def block(name):
        return "\n".join(f"  --{k.replace('_', '-')}: {v};" for k, v in style.TOKENS[name].items())
    return (":root {\n  color-scheme: light;\n" + block("light") + "\n}\n"
            "@media (prefers-color-scheme: dark) {\n:root:not([data-theme=\"light\"]) {\n  color-scheme: dark;\n"
            + block("dark") + "\n}\n}\n:root[data-theme=\"dark\"] {\n  color-scheme: dark;\n" + block("dark") + "\n}\n")


CSS = """
*, *::before, *::after { box-sizing: border-box; border-radius: 0; }
html { background: var(--bg); color: var(--fg);
  font-family: "Berkeley Mono", ui-monospace, "SF Mono", SFMono-Regular, Menlo, "Cascadia Mono", Consolas,
  "DejaVu Sans Mono", monospace; -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); font-size: 0.9375rem; line-height: 1.65; }
.site { max-width: 48rem; margin: 0 auto; padding-block: 4rem 5rem;
  padding-left: max(16px, env(safe-area-inset-left, 0px)); padding-right: max(16px, env(safe-area-inset-right, 0px)); }
.hl { background: var(--hl); padding: 0 0.25em; box-decoration-break: clone; -webkit-box-decoration-break: clone; }
h1 { font-size: 1.375rem; line-height: 1.5; margin: 0 0 0.5rem; }
h2 { font-size: 1.125rem; line-height: 1.4; margin: 3.5rem 0 0.75rem; width: fit-content; padding: 0 0.25em;
  background: var(--hl); }
h3 { font-size: 1rem; margin: 2rem 0 0.5rem; }
.meta { display: inline-block; margin: 0 0 2rem; padding: 0 0.25em; background: var(--box-head); color: var(--muted);
  font-size: 0.875rem; }
.prose { font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; font-size: 1.0625rem;
  line-height: 1.6; }
.prose code { font-family: inherit; font-size: 0.85em; background: var(--hl); padding: 0 0.25em; }
.prose p { margin: 0 0 1.1em; }
.prose dl { display: grid; grid-template-columns: max-content 1fr; gap: 0.3em 1em; margin: 0 0 1.1em; font-size: 0.9375rem; }
.prose dt { font-family: "Berkeley Mono", ui-monospace, Menlo, monospace; font-weight: 700; }
.prose dd { margin: 0; }
a { color: inherit; }
a:hover { background: var(--accent); color: var(--accent-fg); text-decoration: none; }
.theme-toggle { display: flex; width: fit-content; margin: 0 0 1.5rem; background: var(--box-head); font-weight: 700; }
.theme-toggle button { padding: 0 0.4em; border: 0; background: none; color: inherit; font: inherit; cursor: pointer; }
.theme-toggle button[aria-pressed="true"] { background: var(--selected); }
.theme-toggle button:hover { background: var(--accent); color: var(--accent-fg); }
figure.viz { margin: 1.5rem 0 2.25rem; }
figure.viz .frame { background: var(--box); }
figure.viz svg { display: block; width: 100%; height: auto; }
figure.viz .t-dark { display: none; }
:root[data-theme="dark"] figure.viz .t-dark { display: block; }
:root[data-theme="dark"] figure.viz .t-light { display: none; }
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) figure.viz .t-dark { display: block; }
  :root:not([data-theme="light"]) figure.viz .t-light { display: none; }
}
figcaption { margin-top: 0.5em; color: var(--muted); font-family: Inter, system-ui, sans-serif; font-size: 0.875rem;
  line-height: 1.5; }
figcaption b { color: var(--fg); font-family: "Berkeley Mono", ui-monospace, Menlo, monospace; font-weight: 700; }
figcaption .dl { font-family: "Berkeley Mono", ui-monospace, Menlo, monospace; white-space: nowrap; }
.viz-hover use, .viz-hover path { cursor: crosshair; }
.viz-hover use:hover, .viz-hover path:hover { stroke: var(--accent) !important; stroke-width: 2.5px !important; }
.star-head { margin-top: 3rem; }
@media (max-width: 40rem) {
  .prose dl { grid-template-columns: 1fr; gap: 0; }
  .prose dd { margin: 0 0 0.5em; }
  .site { padding-block: 2rem 4rem; }
  figure.viz .frame { overflow-x: auto; }   /* keep figure text readable on phones: pan instead of shrink */
  figure.viz svg { min-width: 36rem; }
}
"""

TOGGLE_JS = """
(function () {
  var root = document.documentElement, box = document.getElementById("theme-toggle");
  if (!box) return;
  function cur() {
    var t = root.dataset.theme;
    if (t) return t;
    return window.matchMedia && matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  function show() {
    box.querySelectorAll("button").forEach(function (b) { b.setAttribute("aria-pressed", String(b.value === cur())); });
  }
  box.hidden = false;
  box.addEventListener("click", function (e) {
    var b = e.target.closest("button"); if (!b) return;
    root.dataset.theme = b.value;
    if (PERSIST) { try { localStorage.setItem("theme", b.value); } catch (err) {} }
    show();
  });
  show();
})();
"""

HEAD_JS = ('try { var t = localStorage.getItem("theme"); if (t === "light" || t === "dark") '
           'document.documentElement.dataset.theme = t; } catch (e) {}')


def inline_svg(path, prefix):
    """An SVG file's markup for inlining: XML header dropped, numbers rounded to 2 decimals, every id and its
    references prefixed so that several SVGs (and both themes of one figure) can share a page."""
    raw = Path(path).read_text()
    raw = re.sub(r"<\?xml[^>]*\?>\s*", "", raw)
    raw = re.sub(r"<!DOCTYPE[^>]*>\s*", "", raw)
    raw = re.sub(r"<metadata>.*?</metadata>\s*", "", raw, flags=re.S)
    ids = set(_ID.findall(raw))

    def ref(m):
        return f'{m.group(1)}#{prefix}{m.group(2)}' if m.group(2) in ids else m.group(0)

    seen = {}

    def rename(m):   # matplotlib names raster images by content, so a repeated image repeats its id
        k = seen[m.group(1)] = seen.get(m.group(1), 0) + 1
        return f'id="{prefix}{m.group(1)}"' if k == 1 else f'id="{prefix}{m.group(1)}-{k}"'

    raw = _ID.sub(rename, raw)
    raw = re.sub(r'(href="|url\()#([^")]+)', ref, raw)
    raw = _NUM.sub(r"\1", raw)
    raw = re.sub(r'<svg([^>]*?) width="[^"]+" height="[^"]+"', r"<svg\1", raw, count=1)
    return raw


def figure_html(entry, n, png_links=False, base=None):
    parts = [f'<figure class="viz" id="{_html.escape(entry.get("anchor", f"fig{n}"))}">', '<div class="frame">']
    for theme in style.THEMES:
        p = entry["paths"].get(theme, {}).get("svg")
        if p:
            parts.append(f'<div class="t-{theme}" role="img" aria-label="{_html.escape(entry["title"])}">'
                         f'{inline_svg(p, f"f{n}{theme[0]}-")}</div>')
    parts.append("</div>")
    cap = f'<b>{_html.escape(entry["title"])}.</b> {_html.escape(entry.get("caption", ""))}'
    if png_links:
        links = []
        for theme in style.THEMES:
            p = entry["paths"].get(theme, {}).get("png")
            if p:
                rel = Path(p).relative_to(base) if base else Path(p).name
                links.append(f'<a href="{_html.escape(str(rel))}">{theme} PNG</a>')
        if links:
            cap += ' <span class="dl">(' + " · ".join(links) + ")</span>"
    parts.append(f"<figcaption>{cap}</figcaption></figure>")
    return "\n".join(parts)


def terms(items, heading="Terms"):
    """A definition list of (term, definition) pairs for a page's intro."""
    rows = "".join(f"<dt>{t}</dt><dd>{d}</dd>" for t, d in items)
    return f"<h3>{heading}</h3><dl>{rows}</dl>"


def page(title, sections, out_path, intro="", meta="", png_links=False, fragment=False):
    """Write the page; returns its path. fragment=True writes it for publishing as a claude.ai Artifact: no
    document skeleton (the host adds it) and the host's theme wins at load (the toggle still works)."""
    out_path = Path(out_path)
    body, n = [], 0
    for sec in sections:
        if sec.get("heading"):
            body.append(f'<h2 class="star-head" id="{_html.escape(sec.get("anchor", ""))}">{_html.escape(sec["heading"])}</h2>')
        if sec.get("text"):
            body.append(f'<div class="prose">{sec["text"]}</div>')
        for entry in sec.get("figures", []):
            body.append(figure_html(entry, n, png_links=png_links, base=out_path.parent))
            n += 1
    head = f"""<title>{_html.escape(title)}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Inter:ital,wght@0,400;0,700;1,400&display=swap">
<style>
{_tokens_css()}{CSS}
</style>"""
    if fragment:
        top = f'{head}\n<meta name="claude:viewport-fit-sides" content="cover">\n'
    else:
        top = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light dark">
<script>{HEAD_JS}</script>
{head}
</head>
<body>
"""
    doc = top + f"""<main class="site">
<h1><span class="hl">{_html.escape(title)}</span></h1>
{f'<p class="meta">{_html.escape(meta)}</p>' if meta else ''}
<div class="theme-toggle" id="theme-toggle" role="group" aria-label="Colour theme" hidden>
<button type="button" value="light">light</button><button type="button" value="dark">dark</button>
</div>
{f'<div class="prose">{intro}</div>' if intro else ''}
{chr(10).join(body)}
</main>
<script>{TOGGLE_JS.replace("PERSIST", "false" if fragment else "true")}</script>
""" + ("" if fragment else "</body>\n</html>\n")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    return out_path
