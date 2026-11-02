"""Heading candidates of a PDF read by LiteParse: every line that could open a section, with what the reader
and the document's own numbering say about it, and the decisions that need no judge: running headers and
footers, table rows and list items are content; a line the reader flagged as a heading that carries the
document's numbering is a section."""
from __future__ import annotations

import difflib
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

NUMBERED = re.compile(r"^(?P<marker>\d{1,2}(?:\.\d{1,2}){0,3}|[IVX]{1,5})\s*[.\-–)]*\s+(?=\S)")
KEYWORD = re.compile(
    r"^(?P<word>CL[ÀA]USULA|CLÁUSULA|ARTICLE|ART[IÍ]CULO|ANNEX(?:O)?|ANEXO|CAP[IÍ]TOL|CAPÍTULO|T[IÍ]TOL|TÍTULO|"
    r"SECCI[OÓ]N?|PART(?:E)?|DISPOSICI[OÓ]N?)\b\.?\s*(?P<num>[0-9IVXLC]{0,6})", re.I)
LETTER_ITEM = re.compile(r"^[a-z]\)\s")
CHAPTER_WORDS = ("capitol", "capitulo", "titol", "titulo", "seccion", "seccio", "part", "parte")
ANNEX_WORDS = ("annex", "annexo", "anexo")
INDEX_PAGE_MIN_ENTRIES = 9
INDEX_PAGE_MIN_LEADERS = 5
REPEAT_MIN_PAGES = 3
MAX_TEXT = 160
SIMILAR = 0.75


@dataclass
class Candidate:
    """A line that could open a section, with the facts a judge weighs."""
    index: int
    page: int
    order: int
    text: str
    md: int = 0
    heading_block: bool = False
    marker: Optional[str] = None
    numbered: bool = False
    keyword: Optional[str] = None
    sources: set = field(default_factory=set)
    repeats: bool = False
    known: bool = False
    decision: Optional[str] = None
    reason: str = ""
    # The text just before and after the line on its page: what a judge reads to tell a heading from a list item.
    previous: str = ""
    following: str = ""
    # The line carries its own page number («3 Departament…» on page 3, «Página 7 de 113» on page 7): a footer.
    page_numbered: bool = False
    # Markdown emphasis around the line (**bold**): a hint the judge weighs; the title itself carries none.
    bold: bool = False
    # Dot leaders and a page number at the end («3. OBJECTE ........ 12»): an entry of the document's index.
    leaders: bool = False

    @property
    def level_hint(self) -> int:
        """1 for a first-level number or a keyword heading, deeper for `1.2`, `1.2.3`; the reader's level otherwise."""
        if self.marker and re.match(r"^\d", self.marker) and self.keyword is None:
            return min(4, self.marker.count(".") + 1)
        if self.keyword:
            return 1
        return min(4, max(1, self.md or 1))


def norm(text: Optional[str]) -> str:
    text = unicodedata.normalize("NFKD", text or "")
    text = "".join(c for c in text if not unicodedata.combining(c)).lower()
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def strip_num(text: Optional[str]) -> str:
    clean = norm(text)
    clean = re.sub(r"^(clausula|article|articulo|annexo|annex|anexo|capitol|capitulo|titol|titulo|seccion|seccio|parte|part|disposicion|disposicio)\s+[0-9ivxlc]*\s*", "", clean)
    return re.sub(r"^(?:\d{1,2}(?:\s\d{1,2}){0,3}|[ivx]{1,5})\s+", "", clean).strip()


def masked(text: Optional[str]) -> str:
    """`Página 7 de 113` and `Página 8 de 113` are one footer: digits masked."""
    return re.sub(r"\d+", "#", norm(text))


def similar(a: str, b: str) -> float:
    na, nb = norm(a), norm(b)
    if not na or not nb:
        return 0.0
    if na == nb:
        return 1.0
    if len(na) >= 12 and (na in nb or nb in na):
        return 0.95
    ratio = difflib.SequenceMatcher(None, na, nb).ratio()
    sa, sb = strip_num(a), strip_num(b)
    if sa and sb:
        ratio = max(ratio, difflib.SequenceMatcher(None, sa, sb).ratio() - 0.05)
    return ratio


def tableish(text: str) -> bool:
    return (text.count("|") >= 2 or len(re.findall(r"\d[\d.,]*\s*(?:€|%)", text)) >= 2
            or len(re.findall(r"\d+[,.]\d+", text)) >= 3)


def marker_of(text: str) -> tuple[Optional[str], bool, Optional[str]]:
    """`(marker, numbered, keyword)`: `1.2` → ("1.2", True, None); `Clàusula 3.` → ("clausula 3", True, "clausula")."""
    keyword = KEYWORD.match(text)
    if keyword:
        word = norm(keyword.group("word"))
        number = keyword.group("num")
        return (f"{word} {number}".strip(), bool(number), word)
    numbered = NUMBERED.match(text)
    if numbered:
        return numbered.group("marker"), True, None
    return None, False, None


def heading_shaped(c: Candidate) -> bool:
    """A numbered line that reads as a heading rather than as an item of a list: short, no bold markers, not
    ending in a colon, a small number, and flagged by the reader, in capitals or in title case."""
    text = c.text
    if c.keyword:
        # «Annex 3», «CAPÍTOL II» or a flagged «ANNEXOS» are headings; «Anexo I de este pliego para cada…» is a sentence.
        if len(text) > MAX_TEXT:
            return False
        if c.numbered or c.heading_block:
            return True
        letters = [ch for ch in text if ch.isalpha()]
        return bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.6 and len(text.split()) <= 8
    if not c.numbered:
        return False
    first = re.match(r"^(\d{1,2})", text)
    if first and int(first.group(1)) > 60:
        return False
    if "**" in text or text.rstrip().endswith(":") or len(text) > 110:
        return False
    body = NUMBERED.sub("", text).strip()
    if len(body) < 4:
        return False
    letters = [ch for ch in body if ch.isalpha()]
    upper_ratio = sum(ch.isupper() for ch in letters) / len(letters) if letters else 0
    words = body.split()
    title_case = sum(w[0].isupper() for w in words if w[0].isalpha()) >= max(1, len(words) // 2)
    return c.heading_block or upper_ratio > 0.6 or (title_case and len(words) <= 8)


def _lines_of(pages: list[dict[str, Any]]) -> list[Candidate]:
    """The first line of every block that could be a heading: heading blocks, and numbered or keyword lines of
    paragraphs and spatial pages. Table rows and bulleted items never are."""
    out: list[Candidate] = []
    for page in pages:
        page_no = int(page.get("page_index") or 0)
        blocks = page.get("blocks") or []
        for order, block in enumerate(blocks):
            kind = block.get("block_type") or ""
            text = (block.get("text") or "").strip()
            if not text or kind == "table":
                continue
            lines = [text.split("\n")[0].strip()] if kind != "page" else [l.strip() for l in text.split("\n") if l.strip()]
            for sub, first in enumerate(lines):
                md = len(re.match(r"^(#*)", first).group(1))
                clean = re.sub(r"^#+\s*", "", first).strip()
                if kind == "list_item":
                    clean = re.sub(r"^[-*•]\s+", "", clean)
                bold = "**" in clean or "__" in clean
                clean = re.sub(r"(\*{1,3}|_{2,3})(?=\S)|(?<=\S)(\*{1,3}|_{2,3})", "", clean).strip()
                leaders = bool(re.search(r"(?:\.\s?){4,}\s*\d{0,4}\s*$", clean))
                marker, numbered, keyword = marker_of(clean)
                heading_block = kind == "heading" or md > 0
                if not heading_block and not numbered and not keyword:
                    continue
                if kind == "list_item" and not re.match(r"^\d", clean):
                    continue
                previous = _snippet(blocks[order - 1].get("text") if order else "", tail=True)
                following = _snippet("\n".join(text.split("\n")[1:]) or (blocks[order + 1].get("text") if order + 1 < len(blocks) else ""))
                page_numbered = _carries_page_number(clean, page_no, keyword)
                out.append(Candidate(index=len(out), page=page_no, order=order * 1000 + sub, text=clean, md=md,
                                     heading_block=heading_block, marker=marker, numbered=numbered,
                                     keyword=keyword, sources={"liteparse"}, previous=previous, following=following,
                                     page_numbered=page_numbered, bold=bold, leaders=leaders))
    return out


def _carries_page_number(text: str, page_no: int, keyword: Optional[str] = None) -> bool:
    """A footer-looking line: one of its first numbers is within five of the physical page («3 Departament…»,
    «Página 7 de 113»; printed numbers often start after a cover). A section number («3.», «1.2», «Cláusula 3») is
    followed by a dot or a bracket, or comes after its word, and never counts."""
    if keyword or re.match(r"^\d{1,4}(?:\.\d{1,3})*\s*[.\-–)]", text):
        return False
    return any(abs(int(n) - page_no) <= 5 for n in re.findall(r"\d+", text)[:3])


def _snippet(text: Optional[str], limit: int = 120, tail: bool = False) -> str:
    flat = re.sub(r"\s+", " ", (text or "")).strip()
    return flat[-limit:] if tail else flat[:limit]


def repeated_texts(cands: Iterable[Candidate]) -> set[str]:
    """Texts seen on three pages or more, with or without their number: running headers, footers, page numbers."""
    counts: dict[str, set[int]] = {}
    for c in cands:
        keys = [norm(c.text), strip_num(c.text)]
        if c.page_numbered:
            keys.append(masked(c.text))  # the same footer with each page's number
        for key in keys:
            if key:
                counts.setdefault(key, set()).add(c.page)
    return {k for k, v in counts.items() if len(v) >= REPEAT_MIN_PAGES}


def is_repeated(c: Candidate, repeated: set[str]) -> bool:
    if norm(c.text) in repeated or strip_num(c.text) in repeated:
        return True
    return c.page_numbered and masked(c.text) in repeated


def noise(c: Candidate, repeated: set[str]) -> Optional[str]:
    """Why a line is content for sure, or None."""
    if is_repeated(c, repeated):
        return "repeated"
    if tableish(c.text):
        return "table"
    if LETTER_ITEM.match(c.text):
        return "letter_item"
    if len(c.text) > MAX_TEXT or len(c.text) < 3:
        return "length"
    if re.match(r"^\d{1,2}[.,]\d{3}\b", c.text):
        return "amount"
    if c.leaders:
        return "index_entry"  # an entry of the index outside its pages: never the section itself
    return None


def index_pages_of(cands: list[Candidate], repeated: set[str]) -> set[int]:
    """Pages listing nine or more headings, or five or more lines with dot leaders: the document's own table of
    contents."""
    shaped: dict[int, int] = {}
    with_leaders: dict[int, int] = {}
    for c in cands:
        if c.leaders:
            with_leaders[c.page] = with_leaders.get(c.page, 0) + 1
        if heading_shaped(c) and noise(c, repeated) is None:
            shaped[c.page] = shaped.get(c.page, 0) + 1
    return ({p for p, n in shaped.items() if n >= INDEX_PAGE_MIN_ENTRIES}
            | {p for p, n in with_leaders.items() if n >= INDEX_PAGE_MIN_LEADERS})


def _located_index_entries(entries: list[Candidate], pages: list[dict[str, Any]], last_index_page: int) -> list[tuple[int, int, str]]:
    """Each index entry at the first later page whose text starts a block with it: `(page, order, title)`."""
    starts: dict[int, list[str]] = {}
    for page in pages:
        no = int(page.get("page_index") or 0)
        if no <= last_index_page:
            continue
        starts[no] = [norm(re.sub(r"^#+\s*", "", (b.get("text") or "").split("\n")[0])) for b in (page.get("blocks") or [])]
    found, seen = [], set()
    for entry in entries:
        had_leaders = bool(re.search(r"\.{3,}", entry.text))
        key = re.sub(r"\s*\.{3,}.*$", "", entry.text).strip()
        if not had_leaders:
            key = re.sub(r"\s{2,}\d{1,3}$", "", key)
        nkey = strip_num(key)[:40] or norm(key)[:40]
        if len(nkey) < 6 or nkey in seen:
            continue
        head = norm(key)[:40]
        for no in sorted(starts):
            hit = next((i for i, t in enumerate(starts[no]) if t.startswith(head) or (strip_num(t) and strip_num(t).startswith(nkey))), None)
            if hit is not None:
                found.append((no, hit * 1000, key))
                seen.add(nkey)
                break
    return found


def _order_in_page(pages: list[dict[str, Any]], page_no: int, title: str) -> int:
    text = norm((pages[page_no - 1].get("markdown") or "") if 0 < page_no <= len(pages) else "")
    i = text.find(norm(title)[:30])
    return i if i >= 0 else 10_000_000


def collect(pages: list[dict[str, Any]], layout_structure: Optional[list[dict]] = None) -> tuple[list[Candidate], dict[str, Any]]:
    """Every candidate of the document, with the noise dropped, the index pages located in the body, the titles
    the layout detection found merged in, and the headings that need no judge marked `known`. Returns the
    candidates in document order and a report of what was dropped."""
    lines = _lines_of(pages)
    repeated = repeated_texts(lines)
    index_pages = index_pages_of(lines, repeated)
    kept: list[Candidate] = []
    dropped = {"repeated": 0, "table": 0, "letter_item": 0, "length": 0, "amount": 0, "index_entry": 0, "index_page": 0}
    for c in lines:
        if c.page in index_pages:
            dropped["index_page"] += 1
            continue
        why = noise(c, repeated)
        if why:
            dropped[why] += 1
            continue
        kept.append(c)
    entries = [c for c in lines if c.page in index_pages and (c.numbered or c.keyword or c.heading_block)]
    for page_no, order, title in _located_index_entries(entries, pages, max(index_pages, default=0)):
        match = _closest(kept, page_no, title)
        if match:
            match.sources.add("toc")
            continue
        marker, numbered, keyword = marker_of(title)
        located = Candidate(index=-1, page=page_no, order=order, text=title, marker=marker, numbered=numbered,
                            keyword=keyword, sources={"toc"}, page_numbered=_carries_page_number(title, page_no, keyword))
        if noise(located, repeated) is None:
            kept.append(located)
    for node in _flatten(layout_structure or []):
        title = node["title"]
        if title.lower().startswith(("page ", "preface")) or title.endswith("(intro)") or len(title) > MAX_TEXT:
            continue
        match = _closest(kept, node["page"], title)
        if match:
            match.sources.add("layout")
            continue
        marker, numbered, keyword = marker_of(title)
        detected = Candidate(index=-1, page=node["page"], order=_order_in_page(pages, node["page"], title), text=title,
                             md=node["depth"], marker=marker, numbered=numbered, keyword=keyword, sources={"layout"},
                             page_numbered=_carries_page_number(title, node["page"], keyword))
        # A footer or a table row the layout took for a title is no better a heading for coming from there.
        if noise(detected, repeated) is None:
            kept.append(detected)
    kept.sort(key=lambda c: (c.page, c.order))
    seen_markers: dict[str, int] = {}
    for i, c in enumerate(kept):
        c.index = i
        if c.marker and re.match(r"^\d", c.marker):
            c.repeats = c.marker in seen_markers
            seen_markers.setdefault(c.marker, i)
        c.known = ("toc" in c.sources or ((c.heading_block or "layout" in c.sources) and (c.numbered or bool(c.keyword))
                                           and heading_shaped(c) and not c.repeats))
        if c.known:
            c.decision, c.reason = "heading", "known"
    return kept, {"lines": len(lines), "candidates": len(kept), "known": sum(c.known for c in kept),
                  "index_pages": sorted(index_pages), "dropped": dropped,
                  "repeated": sorted(k for k in repeated if " " in k)[:20]}


def _closest(kept: list[Candidate], page_no: int, title: str) -> Optional[Candidate]:
    """The candidate the title names: the most similar one on its page, else on the pages next to it."""
    scored = [(c.page == page_no, similar(c.text, title), c) for c in kept if abs(c.page - page_no) <= 1]
    scored = [item for item in scored if item[1] >= SIMILAR]
    return max(scored, key=lambda item: (item[0], item[1]))[2] if scored else None


def _flatten(nodes: list[dict], depth: int = 1, out: Optional[list] = None) -> list[dict]:
    out = [] if out is None else out
    for n in nodes:
        out.append({"title": n.get("title") or "", "page": int(n.get("start_index") or 0), "depth": depth})
        _flatten(n.get("nodes") or [], depth + 1, out)
    return out


def decide_without_judge(c: Candidate) -> str:
    """The deterministic reading of an undecided line: the numbering and the reader's heading flag decide."""
    if c.repeats and not c.heading_block:
        return "body"
    if (c.numbered or c.keyword) and heading_shaped(c):
        return "heading"
    if c.heading_block and c.md <= 2 and 12 <= len(c.text) <= MAX_TEXT and len(c.text.split()) >= 3 \
            and not c.text.rstrip().endswith(":") and "**" not in c.text:
        return "heading"
    return "body"
