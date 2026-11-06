"""Optional LiteParse adapter. Geometry stays in the store, outside LLM prompts.

OCR routing follows rag/packages/liteparse_pipeline: inspect, parse natively,
then replace only the selected physical pages. No dependency on the RAG repo.
"""
from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from time import perf_counter


@dataclass
class PDFExtraction:
    pages: list[dict]
    metadata: dict
    # Seconds per indexing stage, for the caller's log; never stored with the document.
    timings: dict = field(default_factory=dict)

    @property
    def texts(self) -> list[str]:
        return [page["markdown"] for page in self.pages]


def parser_config(value: str | Mapping | None) -> dict:
    """Validate without importing the optional native libraries."""
    conf = {"name": value or "default"} if not isinstance(value, Mapping) else dict(value)
    allowed = {"name", "ocr_mode", "ocr_language", "ocr_server_url", "text_format"}
    if set(conf) - allowed:
        raise ValueError(f"Unknown pdf_parser options: {sorted(set(conf) - allowed)}")
    name = conf.get("name")
    if name == "default" and len(conf) == 1:
        return conf
    if name != "liteparse":
        raise ValueError("pdf_parser must be 'default', 'liteparse', or a dict with name='liteparse'")
    conf = {"ocr_mode": "auto", "ocr_language": "spa", "text_format": "auto", **conf}
    if conf["ocr_mode"] not in ("auto", "enabled", "disabled"):
        raise ValueError("pdf_parser ocr_mode must be auto, enabled or disabled")
    if conf["text_format"] not in ("auto", "markdown", "spatial"):
        raise ValueError("pdf_parser text_format must be auto, markdown or spatial")
    for key in ("ocr_language", "ocr_server_url"):
        if key in conf and not (isinstance(conf[key], str) and conf[key].strip()):
            raise ValueError(f"pdf_parser {key} must be a non-empty string")
    return conf


def _block_text(block: dict) -> str:
    if block.get("kind") == "table":
        header, rows = block.get("header") or [], block.get("rows") or []
        width = max([len(header), *(len(row) for row in rows)], default=0)
        if not width:
            return ""

        def row_text(cells):
            texts = [str(cell.get("text") or "").replace("|", "\\|").replace("\n", "<br>")
                     for cell in cells]
            return "| " + " | ".join(texts + [""] * (width - len(texts))) + " |"

        return "\n".join([row_text(header), "| " + " | ".join(["---"] * width) + " |",
                          *(row_text(row) for row in rows)])
    text = block.get("text") or "\n".join(block.get("lines") or [])
    if block.get("kind") == "heading" and text:
        return "#" * min(6, max(1, block.get("level") or 1)) + " " + text
    if block.get("kind") == "list_item" and text:
        return (block.get("marker") or "-") + " " + text
    return text


def _bbox(rect: dict, width: float, height: float) -> list[float]:
    """Existing SDK block contract: xyxy, top-left, 0..1000 page units."""
    x, y, w, h = (rect[key] for key in ("x", "y", "width", "height"))
    return [max(0.0, min(1000.0, v)) for v in
            (x / width * 1000, y / height * 1000,
             (x + w) / width * 1000, (y + h) / height * 1000)]


def _swallowing_table(block: dict, width: float, height: float) -> bool:
    if block.get("kind") != "table" or not block.get("bbox"):
        return False
    rows = ([block["header"]] if block.get("header") else []) + list(block.get("rows") or [])
    box = block["bbox"]
    area = box.get("width", 0) * box.get("height", 0) / (width * height)
    longest = max((len(str(cell.get("text") or "")) for row in rows for cell in row), default=0)
    return len(rows) <= 4 and area >= 0.5 and longest >= 1200


def _page_record(page: dict, conf: dict, routed: bool) -> dict:
    number, width, height = page["page_num"], page["width"], page["height"]
    if width <= 0 or height <= 0:
        raise ValueError(f"LiteParse returned invalid dimensions on page {number}")
    raw_blocks = page.get("blocks") or []
    spatial = page.get("text") or ""
    # If no table was recognized, keep the parser's column spacing instead
    # of flattening a visibly tabular page into misleading prose. This is
    # representation selection, not a claim to have inferred table cells.
    column_lines = sum(len(re.findall(r"\S[ \t]{3,}(?=\S)", line)) >= 3
                       for line in spatial.splitlines())
    use_spatial = conf["text_format"] == "spatial" or (
        conf["text_format"] == "auto" and column_lines >= 3
        and not any(b.get("kind") == "table" for b in raw_blocks))
    # A page read as one table of a few rows that holds the body in a cell (an e-signature band down the
    # margin makes the reader see two columns): its blocks interleave the text, the spatial text does not.
    use_spatial = use_spatial or (conf["text_format"] == "auto" and any(_swallowing_table(b, width, height)
                                                                         for b in raw_blocks))
    rendered = [(block, _block_text(block)) for block in raw_blocks]
    rendered = [(block, text) for block, text in rendered if text.strip()]
    if use_spatial or not rendered:
        use_spatial = True
        text = spatial.strip("\n")
        if use_spatial and text.strip():
            # Choose a fence that cannot be closed by source content.
            fence = "`" * max(3, 1 + max((len(m[0]) for m in re.finditer(r"`+", text)), default=0))
            text = f"{fence}text\n{text}\n{fence}"
        rendered = [({"kind": "page", "bbox": {"x": 0, "y": 0,
                                                "width": width, "height": height}}, text)] if text.strip() else []
    blocks = []
    for i, (block, text) in enumerate(rendered, 1):
        entry = {"block_id": f"p{number}_text_{i}", "page": number,
                 "block_type": block["kind"], "text": text}
        if block.get("bbox"):
            entry["bbox"] = _bbox(block["bbox"], width, height)
            entry["bbox_points"] = block["bbox"]
        blocks.append(entry)
    return {"page_index": number,
            "markdown": "\n\n".join(block["text"] for block in blocks),
            "citation_markdown": "\n\n".join(
                f"[block_id={block['block_id']}]\n{block['text']}" for block in blocks),
            "blocks": blocks,
            "extraction": {"parser": "liteparse", "ocr_routed": routed,
                           "ocr_text_present": any(t.get("confidence") is not None
                                                   for t in page.get("text_items", [])),
                           "text_format": "spatial" if use_spatial else "blocks"},
            "layout": {"coordinate_system": {"origin": "top-left", "units": "pt",
                                               "dpi": 72, "space": "displayed_page"},
                       **page}}


def _inspector_headings(inspector, data: bytes, count: int) -> list[list[dict]]:
    """The headings pdf-inspector's own reading finds from the typography (its markdown «#» lines), per page: an
    opinion independent of LiteParse's, for the native pages. Unavailable or failing: []."""
    import re as _re
    try:
        result = inspector.extract_pages_markdown_bytes(data)
    except Exception:
        return [[] for _ in range(count)]
    out: dict[int, list[dict]] = {}
    for page in getattr(result, "pages", None) or []:
        heads = []
        for line in (getattr(page, "markdown", "") or "").splitlines():
            m = _re.match(r"^(#{1,6})\s+(.+)$", line.strip())
            if m:
                heads.append({"level": len(m.group(1)), "text": _re.sub(r"[*_]{2,}", "", m.group(2)).strip()[:300]})
        out[int(getattr(page, "page", 0) or 0)] = heads
    return [out.get(n, []) for n in range(1, count + 1)]


def _tagged_headings(parser_class, data: bytes, count: int) -> list[list[dict]]:
    """The headings a tagged PDF declares (H1-H6 in its structure tree, as Word exports its heading styles), per
    page, with their level and text. A separate light pass: the text metadata that links tags to text changes how
    text items are split, so the stored text and geometry stay those of the main pass. Untagged or unreadable: []."""
    import re as _re
    try:
        result = asdict(parser_class(extract_structure_tree=True, extract_text_metadata=True, extract_blocks=False,
                                     output_format="text", ocr_enabled=False, quiet=True,
                                     continue_on_page_error=True).parse(data))
    except Exception:
        return [[] for _ in range(count)]
    out: dict[int, list[dict]] = {}
    for page in result.get("pages") or []:
        heads: list[tuple[int, list]] = []

        def walk(node: dict, inside: bool = False) -> None:
            kind = node.get("type") or ""
            level = int(kind[1]) if _re.fullmatch(r"H[1-6]", kind) else (1 if kind == "H" else 0)
            if level and not inside:
                ids: list = []

                def collect(n: dict) -> None:
                    ids.extend(n.get("marked_content_ids") or [])
                    for child in n.get("children") or []:
                        collect(child)
                collect(node)
                heads.append((level, ids))
                return
            for child in node.get("children") or []:
                walk(child, inside)

        for root in (page.get("structure_tree") or {}).get("roots") or []:
            walk(root)
        found = []
        for level, ids in heads:
            wanted = set(ids)
            items = sorted((t for t in page.get("text_items") or [] if t.get("mcid") in wanted),
                           key=lambda t: (round(t["y"]), t["x"]))
            text = _re.sub(r"\s+", " ", " ".join(t.get("text") or "" for t in items)).strip()
            if text:
                found.append({"level": level, "text": text[:300]})
        out[page.get("page_num")] = found
    return [out.get(n, []) for n in range(1, count + 1)]


def extract_liteparse(file_path: str, config: str | Mapping) -> PDFExtraction:
    conf = parser_config(config)
    try:
        from liteparse import LiteParse
        import pdf_inspector
    except ImportError as exc:
        raise ImportError('LiteParse requires the optional extra: pip install "pageindex[liteparse]"') from exc
    started = perf_counter()
    data = Path(file_path).read_bytes()
    mode, routed, inspection = conf["ocr_mode"], [], None
    if mode == "auto":
        detected = pdf_inspector.detect_pdf_bytes(data)
        routed = sorted(set(detected.pages_needing_ocr))
        inspection = {"version": version("pdf-inspector"), "page_count": detected.page_count,
                      "pdf_type": detected.pdf_type,
                      "ocr_reasons_by_page": [{"page": r.page, "reasons": list(r.reasons)}
                                              for r in detected.ocr_reasons_by_page]}
    passes = []

    def parse(ocr: bool, targets: list[int] | None = None) -> dict:
        parser = LiteParse(ocr_enabled=ocr, ocr_language=conf["ocr_language"],
                           ocr_server_url=conf.get("ocr_server_url"),
                           target_pages=",".join(map(str, targets)) if targets else None,
                           output_format="markdown", keep_headers_footers=True,
                           extract_blocks=True, emit_word_boxes=True,
                           extract_images=False, extract_screenshots=False,
                           extract_content_bounds=True, preserve_very_small_text=True,
                           continue_on_page_error=False, ocr_failure_fatal=True, quiet=True)
        tick = perf_counter()
        result = asdict(parser.parse(data))
        if result.get("page_errors"):
            raise ValueError(f"LiteParse page errors: {result['page_errors']}")
        passes.append({"ocr_enabled": ocr, "target_pages": targets,
                       "seconds": perf_counter() - tick})
        return result

    all_ocr = mode == "enabled" or (inspection is not None and len(routed) == inspection["page_count"])
    result = parse(all_ocr)
    count = result["total_pages"]
    if mode == "enabled":
        routed = list(range(1, count + 1))
    by_page = {page["page_num"]: page for page in result["pages"]}
    if len(result["pages"]) != count or set(by_page) != set(range(1, count + 1)):
        raise ValueError("LiteParse returned missing or duplicate physical pages")
    if inspection is not None and inspection["page_count"] != count:
        raise ValueError("LiteParse and pdf-inspector disagree on the physical page count")
    if not set(routed) <= set(by_page):
        raise ValueError("pdf-inspector selected pages outside the PDF")
    if routed and not all_ocr:
        targeted = parse(True, routed)
        replacement = {page["page_num"]: page for page in targeted["pages"]}
        if targeted["total_pages"] != count or set(replacement) != set(routed) or len(replacement) != len(targeted["pages"]):
            raise ValueError("LiteParse OCR pass did not return exactly the requested physical pages")
        by_page.update(replacement)
    pages = [_page_record(by_page[n], conf, n in routed) for n in range(1, count + 1)]
    for record, heads in zip(pages, _tagged_headings(LiteParse, data, count)):
        if heads and record["page_index"] not in routed:
            record["tagged_headings"] = heads
    for record, heads in zip(pages, _inspector_headings(pdf_inspector, data, count)):
        if heads and record["page_index"] not in routed:
            record["inspector_headings"] = heads
    return PDFExtraction(pages, {"parser": "liteparse", "version": version("liteparse"),
                                "ocr_mode": mode, "ocr_language": conf["ocr_language"],
                                "ocr_backend": "http" if conf.get("ocr_server_url") else "tesseract",
                                "ocr_pages": routed, "inspection": inspection,
                                "ocr_empty_pages": [n for n in routed if not (by_page[n].get("text") or "").strip()],
                                "passes": passes, "seconds": perf_counter() - started})


def locate_text(pages: list[dict], quote: str) -> list[dict]:
    """Exact word-stream matches, normalizing Unicode and whitespace only.

    Return every occurrence, never choose a repeated phrase arbitrarily. No
    fuzzy paraphrase matching: a missing or discontinuous quote has no boxes.
    """
    if not isinstance(quote, str) or not quote.strip():
        raise ValueError("quote must be a non-empty string")

    def normalize(text):
        return " ".join(unicodedata.normalize("NFKC", text).split())

    target, matches = normalize(quote), []
    for page in pages:
        layout = page.get("layout") or {}
        words, spans, stream = [], [], ""
        for item in layout.get("text_items") or []:
            text = normalize(item.get("text") or "")
            if not text:
                continue
            start = len(stream) + bool(stream)
            stream += (" " if stream else "") + text
            # Some rotated text has per-glyph rather than per-word boxes.
            # Anchor each box to the item's actual text; inserting spaces
            # between every returned box would turn "Hello" into "H e l l o".
            anchored, cursor = [], 0
            for word in item.get("words") or [item]:
                token = normalize(word.get("text") or "")
                pos = text.find(token, cursor) if token else -1
                if pos < 0 or text[cursor:pos].strip():
                    anchored = [(item, 0, len(text))]
                    cursor = len(text)
                    break
                anchored.append((word, pos, pos + len(token)))
                cursor = pos + len(token)
            if text[cursor:].strip():
                anchored = [(item, 0, len(text))]
            for word, a, b in anchored:
                words.append(word)
                spans.append((start + a, start + b))
        offset = 0
        while (start := stream.find(target, offset)) >= 0:
            end = start + len(target)
            selected = [word for word, (a, b) in zip(words, spans) if a < end and b > start]
            matches.append({"page": page["page_index"], "text": stream[start:end],
                            "match": "exact_normalized", "coordinate_system": "top-left-0-1000",
                            "boxes": [_bbox(w, layout["width"], layout["height"]) for w in selected],
                            "boxes_points": [{k: w[k] for k in ("text", "x", "y", "width", "height")}
                                             for w in selected]})
            # Include overlapping occurrences too ("a a" in "a a a").
            offset = start + 1
    return matches
