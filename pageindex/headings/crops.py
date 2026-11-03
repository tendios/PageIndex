"""A crop of the page around a candidate line, for the judge: the line framed in red with a few lines of context,
rendered from the PDF with PDFium and sized within TypeSafe's limits (each side 280 to 1568 pixels)."""
from __future__ import annotations

import re
import threading
import unicodedata
from io import BytesIO
from typing import Any, Optional

MIN_SIDE, MAX_SIDE = 280, 1568
SCALE = 2.0                      # 144 dpi
_lock = threading.Lock()
_docs: dict[str, Any] = {}


def _norm(text: str) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    return re.sub(r"\s+", " ", "".join(ch for ch in text if not unicodedata.combining(ch)).lower()).strip()


def _line_y(page: dict[str, Any], text: str) -> Optional[tuple[float, float]]:
    """Top and height, in points, of the first text item that starts the line's words."""
    layout = page.get("layout") or {}
    items = sorted((t for t in layout.get("text_items") or [] if (t.get("text") or "").strip()),
                   key=lambda t: (round(t["y"]), t["x"]))
    words = _norm(re.sub(r"[*_#|]+", " ", text)).split()[:3]
    if not words:
        return None
    key = " ".join(words)
    for k, t in enumerate(items):
        joined = _norm(" ".join(x["text"] for x in items[k:k + 4]))
        if joined.startswith(key) or (len(words[0]) > 2 and _norm(t["text"]).startswith(words[0]) and key in joined[:80]):
            return float(t["y"]), float(t.get("height") or 10)
    return None


_rendered: "OrderedDict[tuple[str, int], Any]" = None  # type: ignore[assignment]
CACHE_PAGES = 12


def _page_image(pdf_path: str, page_no: int):
    """The page rendered once and kept for its next lines (asked lines come in document order)."""
    global _rendered
    import pypdfium2 as pdfium
    from collections import OrderedDict
    with _lock:
        if _rendered is None:
            _rendered = OrderedDict()
        key = (pdf_path, page_no)
        if key in _rendered:
            _rendered.move_to_end(key)
            return _rendered[key]
        doc = _docs.get(pdf_path)
        if doc is None:
            doc = _docs[pdf_path] = pdfium.PdfDocument(pdf_path)
        image = doc[page_no - 1].render(scale=SCALE).to_pil().convert("RGB")
        _rendered[key] = image
        while len(_rendered) > CACHE_PAGES:
            _rendered.popitem(last=False)
        return image


def line_crop(pdf_path: str, pages: list[dict[str, Any]], c: Any) -> Optional[bytes]:
    try:
        import pypdfium2 as pdfium
        from PIL import Image, ImageDraw
    except ImportError:
        return None
    if not 0 < c.page <= len(pages):
        return None
    page = pages[c.page - 1]
    layout = page.get("layout") or {}
    width, height = float(layout.get("width") or 0), float(layout.get("height") or 0)
    found = _line_y(page, c.text)
    if not found or not width or not height:
        return None
    y, h = found
    top, bottom = max(0.0, y - 3.2 * h), min(height, y + 4.5 * h)
    rendered = _page_image(pdf_path, c.page)
    sx, sy = rendered.width / width, rendered.height / height
    crop = rendered.crop((0, int(top * sy), rendered.width, int(bottom * sy)))
    draw = ImageDraw.Draw(crop)
    draw.rectangle((int(0.02 * crop.width), int((y - top - 0.25 * h) * sy), int(0.98 * crop.width),
                    int((y - top + 1.25 * h) * sy)), outline=(220, 0, 0), width=3)
    scale = min(MAX_SIDE / max(crop.size), 1.0)
    if scale < 1.0:
        crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))))
    canvas = Image.new("RGB", (max(crop.width, MIN_SIDE), max(crop.height, MIN_SIDE)), "white")
    canvas.paste(crop, ((canvas.width - crop.width) // 2, (canvas.height - crop.height) // 2))
    out = BytesIO()
    canvas.save(out, "PNG", optimize=True)
    return out.getvalue()
