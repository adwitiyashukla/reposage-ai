"""Generate the architecture diagram as SVG.

GitHub's Mermaid renderer is version-pinned and occasionally fails on diagrams
that parse cleanly everywhere else, which leaves a broken box where the most
important picture in the README should be. A committed SVG always renders, on
GitHub and anywhere the README is mirrored.

Two files are emitted from one layout so the light and dark variants can never
drift apart. Regenerate with:

    python docs/assets/make_architecture_svg.py
"""

from __future__ import annotations

from pathlib import Path

W, H = 1000, 575

LIGHT = {
    "band": "#f6f8fa", "band_stroke": "#d0d7de", "band_label": "#57606a",
    "box": "#ffffff", "box_stroke": "#d0d7de", "text": "#1f2328", "sub": "#57606a",
    "accent": "#ddf4ff", "accent_stroke": "#54aeff",
    "store": "#fff8c5", "store_stroke": "#d4a72c",
    "out": "#dafbe1", "out_stroke": "#2da44e",
    "arrow": "#6e7781", "loop": "#cf222e",
}
DARK = {
    "band": "#161b22", "band_stroke": "#30363d", "band_label": "#8b949e",
    "box": "#0d1117", "box_stroke": "#30363d", "text": "#e6edf3", "sub": "#8b949e",
    "accent": "#0d2d6b", "accent_stroke": "#316dca",
    "store": "#3a2a00", "store_stroke": "#9e6a03",
    "out": "#0f2f1a", "out_stroke": "#2ea043",
    "arrow": "#8b949e", "loop": "#f85149",
}

FONT = "-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif"
MONO = "ui-monospace,SFMono-Regular,'SF Mono',Menlo,Consolas,monospace"


def esc(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def band(x, y, w, h, label, p):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="10" '
        f'fill="{p["band"]}" stroke="{p["band_stroke"]}" stroke-width="1"/>'
        f'<text x="{x + 16}" y="{y + 22}" font-family="{MONO}" font-size="11" '
        f'font-weight="600" letter-spacing="1.2" fill="{p["band_label"]}">{esc(label)}</text>'
    )


def box(x, y, w, h, title, subtitle, p, kind="box"):
    fill, stroke = {
        "box": (p["box"], p["box_stroke"]),
        "accent": (p["accent"], p["accent_stroke"]),
        "store": (p["store"], p["store_stroke"]),
        "out": (p["out"], p["out_stroke"]),
    }[kind]
    cx = x + w / 2
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
        f'stroke="{stroke}" stroke-width="1.5"/>'
    ]
    ty = y + h / 2 + (-4 if subtitle else 5)
    out.append(
        f'<text x="{cx}" y="{ty}" text-anchor="middle" font-family="{FONT}" '
        f'font-size="13.5" font-weight="600" fill="{p["text"]}">{esc(title)}</text>'
    )
    if subtitle:
        out.append(
            f'<text x="{cx}" y="{ty + 16}" text-anchor="middle" font-family="{FONT}" '
            f'font-size="11" fill="{p["sub"]}">{esc(subtitle)}</text>'
        )
    return "".join(out)


def arrow(x1, y1, x2, y2, p, dashed=False, colour=None, label=None, lx=None, ly=None):
    stroke = colour or p["arrow"]
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    marker = "urlloop" if colour == p["loop"] else "url"
    out = [
        f'<path d="M {x1} {y1} L {x2} {y2}" fill="none" stroke="{stroke}" '
        f'stroke-width="1.6"{dash} marker-end="url(#arrow-{"loop" if colour == p["loop"] else "std"})"/>'
    ]
    _ = marker
    if label:
        out.append(
            f'<text x="{lx}" y="{ly}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="10.5" fill="{stroke}">{esc(label)}</text>'
        )
    return "".join(out)


def elbow(points, p, colour=None, label=None, lx=None, ly=None, dashed=False):
    stroke = colour or p["arrow"]
    dash = ' stroke-dasharray="5,4"' if dashed else ""
    d = f"M {points[0][0]} {points[0][1]} " + " ".join(f"L {x} {y}" for x, y in points[1:])
    out = [
        f'<path d="{d}" fill="none" stroke="{stroke}" stroke-width="1.6"{dash} '
        f'marker-end="url(#arrow-{"loop" if colour == p["loop"] else "std"})"/>'
    ]
    if label:
        out.append(
            f'<text x="{lx}" y="{ly}" text-anchor="middle" font-family="{MONO}" '
            f'font-size="10.5" fill="{stroke}">{esc(label)}</text>'
        )
    return "".join(out)


def render(p: dict[str, str]) -> str:
    s: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" font-family="{FONT}">',
        "<defs>",
        f'<marker id="arrow-std" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{p["arrow"]}"/></marker>',
        f'<marker id="arrow-loop" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse">'
        f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{p["loop"]}"/></marker>',
        "</defs>",
    ]

    # ---------------------------------------------------------------- bands
    # Bands stop short of the right edge, leaving a routing channel at x=945
    # for the long feedback edge that would otherwise cut through them.
    s.append(band(20, 20, 900, 130, "INGESTION", p))
    s.append(band(20, 172, 900, 130, "INDEX", p))
    s.append(band(20, 324, 900, 225, "AGENT", p))

    # ------------------------------------------------------------ ingestion
    y = 62
    s.append(box(40, y, 160, 62, "Git repository", "clone or local path", p))
    s.append(box(240, y, 185, 62, "Walker", "gitignore, binaries, budget", p))
    s.append(box(465, y, 200, 62, "AST chunker", "one chunk per declaration", p))
    s.append(box(705, y, 175, 62, "Repository map", "symbol-annotated tree", p))
    for x1, x2 in ((200, 240), (425, 465), (665, 705)):
        s.append(arrow(x1, y + 31, x2 - 4, y + 31, p))

    # ---------------------------------------------------------------- index
    yi = 214
    s.append(box(240, yi, 185, 62, "Gemini embeddings", "cached, quota-shaped", p))
    s.append(box(465, yi, 200, 62, "Vector store", "exact cosine, float32", p, "store"))
    s.append(box(705, yi, 175, 62, "BM25", "code-aware tokens", p, "store"))
    s.append(arrow(425, yi + 31, 461, yi + 31, p))
    s.append(elbow([(565, 124), (565, 160), (332, 160), (332, yi - 4)], p))
    s.append(elbow([(565, 124), (565, 160), (792, 160), (792, yi - 4)], p))

    # ---------------------------------------------------------------- agent
    ya = 366
    s.append(box(40, ya, 140, 58, "Question", None, p))
    s.append(box(215, ya, 140, 58, "Plan", "decompose", p, "accent"))
    s.append(box(390, ya, 150, 58, "Retrieve", "dense + lexical", p, "accent"))
    s.append(box(575, ya, 150, 58, "Fuse + rerank", "RRF, then LLM", p, "accent"))
    s.append(box(760, ya, 120, 58, "Analyse", "draft + cite", p, "accent"))
    for x1, x2 in ((180, 215), (355, 390), (540, 575), (725, 760)):
        s.append(arrow(x1, ya + 29, x2 - 4, ya + 29, p))

    yb = 476
    s.append(box(560, yb, 160, 58, "Critique", "grounded? complete?", p, "accent"))
    s.append(box(310, yb, 210, 58, "Verify citations", "resolve, score confidence", p, "accent"))
    s.append(box(40, yb, 230, 58, "Answer", "citations, cost, trace", p, "out"))

    s.append(elbow([(820, 424), (820, 505), (724, 505)], p))
    s.append(arrow(560, yb + 29, 524, yb + 29, p, label="accept", lx=540, ly=yb + 18))
    s.append(arrow(310, yb + 29, 274, yb + 29, p))
    s.append(
        elbow(
            [(640, yb), (640, 452), (465, 452), (465, ya + 62)],
            p, colour=p["loop"], label="refine", lx=553, ly=447,
        )
    )
    # The planner reads the repository map before writing any query.
    s.append(
        elbow(
            [(880, 93), (945, 93), (945, 344), (285, 344), (285, ya - 4)],
            p, dashed=True, label="repository map", lx=700, ly=338,
        )
    )
    # Both indexes feed retrieval.
    s.append(elbow([(565, 276), (565, 308), (430, 308), (430, ya - 4)], p))
    s.append(elbow([(792, 276), (792, 316), (500, 316), (500, ya - 4)], p))

    s.append("</svg>")
    return "".join(s)


def main() -> None:
    here = Path(__file__).parent
    for name, palette in (("architecture-light.svg", LIGHT), ("architecture-dark.svg", DARK)):
        target = here / name
        target.write_text(render(palette), encoding="utf-8")
        print(f"wrote {target.relative_to(here.parent.parent)} ({target.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
