"""Lint the single-page front end against the author's look requirements (FR-9, FR-11)."""
import re
from pathlib import Path

PAGE = (Path(__file__).resolve().parent.parent / "src/kinect3dgs/ui/static/index.html").read_text()


def lum(hexcol: str) -> float:
    h = hexcol.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = [int(h[i:i + 2], 16) / 255 for i in (0, 2, 4)]
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def contrast(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True)
    return (la + 0.05) / (lb + 0.05)


def css():
    return PAGE.split("<style", 1)[1].split("</style>", 1)[0]


def test_no_colour_brighter_than_e0e0e0_and_palette_is_dark_and_muted():
    cols = set(re.findall(r"#[0-9a-fA-F]{6}\b|#[0-9a-fA-F]{3}\b", css()))
    assert cols, "no colours found"
    for c in cols:
        assert lum(c) <= lum("#e0e0e0") + 1e-9, f"{c} is brighter than #e0e0e0"
    assert lum("#0d0d0d") < 0.01                                           # background is near black


def test_text_contrast_readable_but_muted():
    assert contrast("#a8a8a8", "#0d0d0d") >= 7                             # body text
    assert contrast("#cfcfcf", "#0d0d0d") >= 7                             # headings
    assert contrast("#6a6a6a", "#0d0d0d") >= 3                             # dim hints only
    for c in ("#6f8f72", "#a89560", "#a86f6f"):                            # status colours: readable, not loud
        assert contrast(c, "#0d0d0d") >= 4.5


def test_no_decoration():
    c = css()
    for forbidden in ("border-radius", "box-shadow", "text-shadow", "gradient", "transition", "animation", "@keyframes", "@font-face", "filter:"):
        assert forbidden not in c, forbidden
    assert "<svg" not in PAGE and "<canvas" not in PAGE and "<link" not in PAGE


def test_no_external_requests_and_no_inline_style_attributes():
    urls = re.findall(r"https?://[^\s\"'<>)]+", PAGE)
    assert urls == [], urls
    assert not re.findall(r'\sstyle="', PAGE)                                # CSP blocks them
    assert not re.findall(r'\son[a-z]+="', PAGE)                             # no inline event handlers
    assert 'nonce="__NONCE__"' in PAGE and PAGE.count("nonce=") == 2
    assert "__TOKEN__" in PAGE


def test_monospace_single_column_and_keys_documented():
    assert "monospace" in css() and "max-width:720px" in css().replace(" ", "")
    for key in ("[space]", "[n]", "[f]", "[d]", "[p]", "[esc]", "[r]", "[enter]", "[c]", "[o]"):
        assert key in PAGE
