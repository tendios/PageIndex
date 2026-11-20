"""Each native page of a LiteParse read checked against PDFium's text layer, and the lines the read lost put back.

LiteParse (2.15 to 2.20 at least) sometimes drops whole lines when it reads without text metadata, as the fork
reads: a glyph whose box is far taller than its type (a hyphen set in Cambria Math has a 56 pt box for 10 pt text)
swallows the lines around it, and tight table rows under tall item boxes lose cells. Nothing in the read says so.

Per page that is not read with OCR, after the read:

1. The check. PDFium's words (pypdfium2, the text layer in one call, about a millisecond a page) against the read's
   text and blocks, whitespace and line-end hyphens normalised. Left out: lines rotated against the page's own
   direction (a signature band down the margin), text under 3 pt, signature and verification-code lines (CSV), and
   text LiteParse hides on purpose because a clip path hides it. A word is covered when a three-word run holding it
   is in the read; a run of MIN_RUN or more uncovered words the read lacks by count and by letters is lost. Only a
   page with a lost run is read again with PDFium's per-character geometry, and checked again with it.
2. The recovery. Each lost run, its consecutive lost lines together, goes back where it was, from its neighbours'
   words in PDFium's order (the run's anchors) and its lines' positions: inside the block whose text holds both
   anchors side by side, at the end of the block before it or the start of the block after it (by the line pitch:
   no paragraph gap on that side), or as a block of its own between them. The same place is found in the page's
   blocks, its markdown and its text, and the run's lines are added to its text items, so every view of the page
   holds the same text; a run any of them cannot place is not placed at all. A table cell never is: the read's grid
   moved around the lost row.
3. The fallback, for a page with lost text that could not be placed: the page alone read again with the text
   metadata (extract_text_metadata), and then, if that leaves some still lost and OCR is allowed, with OCR too; each
   read is checked and recovered the same way. The page keeps the read that lacks the fewest PDFium words
   (`missing`), among those whose structure is not worse than the first read's (`worse_structure`: no prose moved
   into tables, no page swallowed by one table, no prose cut into fragments, no blocks turned into spatial text); a
   tie keeps the first read.

Recovered text is marked: a text item made from PDFium's line has ``recovered: True`` and no font metadata (the
fields LiteParse leaves empty when it does not read them); a block holding recovered text, and the page's text and
markdown, list the spans of their text that are recovered with what each replaced (``recovered``), so the read as
LiteParse gave it can be rebuilt (``as_read``). The headings mode reads the pages that way: recovered lines are
never heading candidates and never move one.

No model call and no setting: the check is part of the read.
"""
from __future__ import annotations

import ctypes
import re
import statistics
import threading
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable, Optional

MIN_RUN = 5                 # a lost run: this many PDFium words or more
TINY = 3.0                  # text under this many points is left out
PDFIUM_LOCK = threading.Lock()   # PDFium is not thread-safe: the check and the headings' crops hold it

_TRANS = str.maketrans({"“": '"', "”": '"', "«": '"', "»": '"', "‘": "'", "’": "'", "‐": "-", "‑": "-",
                        "–": "-", "—": "-", "−": "-", "\u00ad": "", "\ufffe": "", "\u00a0": " "})
_WORD = re.compile(r"[^\W_]+")
_RAW_WORD = re.compile("(?:[^\\W_]|[\u00ad\ufffe])+")
# Lines of a signature, a verification code or a CSV footer: PDFium reads them where LiteParse may not.
_ESIGN = re.compile(r"codi validaci|esp[úu]blico|gestiona\b|c[óo]digo seguro de verificaci|codi segur de verificaci|"
                    r"\bcsv\b|url de verificaci|verificaci[óo]n.{0,40}https?:|validar.{0,40}https?:|"
                    r"signat electr[òo]nicament per|firmado electr[óo]nicamente por|firmado digitalmente|"
                    r"signat digitalment|firmat per|signat per|powered by tcpdf", re.I)
_CODE = re.compile(r"\b[A-Z0-9]{20,}\b")
_HYPHEN_END = re.compile(r"[^\W\d_]-$")


def _norm(text: str) -> str:
    return unicodedata.normalize("NFKC", text or "").translate(_TRANS).lower()


_ONE_TO_ONE = re.compile("[\u00ad\ufffe\u0130]")   # what _norm drops, or lowers into two characters


def _spans(text: str) -> list[tuple[str, int, int]]:
    """The words of a text, normalised, each with the span of the text it comes from."""
    text = text or ""
    if unicodedata.is_normalized("NFKC", text) and not _ONE_TO_ONE.search(text):
        # _norm maps this text character for character, and its words untouched but for their case (it changes
        # only quotes, dashes and spaces): its words are where they are in the text
        return [(m.group(), m.start(), m.end()) for m in _WORD.finditer(text.lower())]
    out = []
    for m in _RAW_WORD.finditer(text):
        for token in _WORD.findall(_norm(m.group())):
            out.append((token, m.start(), m.end()))
    return out


def _tokens(text: str) -> list[str]:
    text = text or ""
    if unicodedata.is_normalized("NFKC", text) and not _ONE_TO_ONE.search(text):
        return _WORD.findall(text.lower())
    return [t for t, _, _ in _spans(text)]


def _dehyph(text: str) -> str:
    return re.sub(r"(\w)-[ \t]*\n[ \t]*(\w)", r"\1\2", text or "")


def _letter(ch: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", ch) if c.isalnum() and not unicodedata.combining(c)).lower()


def _letters(text: str) -> tuple[str, list[int]]:
    """The text's letters and digits alone, accents and case dropped, with the index in the text of each one."""
    out, where = [], []
    for k, ch in enumerate(text or ""):
        if ch.isspace():
            continue
        for c in _letter(ch):
            out.append(c)
            where.append(k)
    return "".join(out), where


def _block_text(block: dict) -> str:
    from .pdf_extraction import _block_text as render
    return render(block)


# ── what a read holds ────────────────────────────────────────────────────────────────────────────────────────


class _Read:
    """The words a read holds on a page, in its text and its blocks (as the fork renders them)."""

    def __init__(self, page: dict):
        text = page.get("text") or ""
        joined = "\n".join(_block_text(b) for b in page.get("blocks") or [])
        streams = []
        for t in (text, joined):
            plain, dehyphenated = _tokens(t), _dehyph(t)
            streams += [plain, plain if dehyphenated == t else _tokens(dehyphenated)]
        self.grams: set = set()
        self.bigrams: set = set()
        for k, s in enumerate(streams):
            if k % 2 and s is streams[k - 1]:
                continue  # no line-end hyphen: the same words
            self.grams.update(zip(s, s[1:], s[2:]))
            self.bigrams.update(zip(s, s[1:]))
        self.counts = Counter(streams[0]) | Counter(streams[2])
        self.joined = ["".join(s) for s in streams]
        self.flat = self.joined[0] + "|" + self.joined[2] + "|" + self.joined[1]

    def covered(self, tokens: list[str]) -> list[bool]:
        n = len(tokens)
        if n == 1:
            return [tokens[0] in self.counts]
        if n == 2:
            return [tuple(tokens) in self.bigrams] * 2
        flags = [False] * n
        for i in range(n - 2):
            if (tokens[i], tokens[i + 1], tokens[i + 2]) in self.grams:
                flags[i] = flags[i + 1] = flags[i + 2] = True
        return flags

    def letters_found(self, tokens: list[str], size: int = 20, step: int = 5) -> float:
        s = "".join(tokens)
        if not s:
            return 1.0
        if len(s) <= size:
            return 1.0 if s in self.flat else 0.0
        chunks = [s[i:i + size] for i in range(0, len(s) - size + 1, step)]
        return sum(c in self.flat for c in chunks) / len(chunks)

    def deficit(self, tokens: list[str], reference: Counter) -> float:
        if not tokens:
            return 0.0
        need = Counter(tokens)
        return sum(min(k, max(0, reference[w] - self.counts[w])) for w, k in need.items()) / len(tokens)

    def lacks(self, parts: list[list[str]], reference: Counter) -> bool:
        """Whether these words (split by PDFium line) are missing from the read, not merely respaced or reordered:
        under half their letters are there, and the read has at least 60 % fewer of them than PDFium."""
        flat = [t for p in parts for t in p]
        if not flat:
            return False
        weights = sum(len(p) for p in parts)
        found = sum(self.letters_found(p) * len(p) for p in parts if p) / weights
        return found < 0.5 and self.deficit(flat, reference) >= 0.6


# ── PDFium's lines ───────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Line:
    text: str
    excluded: bool = False
    bbox: Optional[tuple[float, float, float, float]] = None   # displayed page space, glyph boxes
    baseline: Optional[float] = None
    size: float = 0.0
    boxes: list = field(default_factory=list)                  # per character of text: its box or None

    def __post_init__(self):
        self.tokens = _tokens(self.text)

    @property
    def spans(self) -> list[tuple[str, int, int]]:
        return _spans(self.text)


def _quick_lines(text: str) -> list[_Line]:
    """PDFium's text of a page, in one call, as lines: enough to tell a clean page."""
    lines = []
    for raw in re.split(r"\r\n|\r|\n", text or ""):
        if raw.strip():
            lines.append(_Line(raw, excluded=bool(_ESIGN.search(raw) or _CODE.search(raw))))
    return lines


def _angle_delta(a: float, b: float) -> float:
    return abs((a - b + 180) % 360 - 180)


def _geometry_lines(pdf_page) -> list[_Line]:
    """PDFium's lines of a page with their characters' boxes, in the displayed page space LiteParse uses (top-left
    origin, the page's rotation applied): a line ends at a line break, at a change of direction, or at a gap wider
    than two and a half times the type."""
    import math
    import pypdfium2.raw as pdfium_c
    textpage = pdf_page.get_textpage()
    try:
        width, height = pdf_page.get_size()
        raw, page_raw = textpage.raw, pdf_page.raw
        scale = 20
        ix, iy = ctypes.c_int(), ctypes.c_int()

        def device(x: float, y: float) -> tuple[float, float]:
            pdfium_c.FPDF_PageToDevice(page_raw, 0, 0, int(width * scale), int(height * scale), 0, x, y,
                                       ctypes.byref(ix), ctypes.byref(iy))
            return ix.value / scale, iy.value / scale

        # PDFium's page-to-display transform is affine: three points give it
        k = 1000.0
        ox, oy = device(0, 0)
        ax, ay = device(k, 0)
        bx, by = device(0, k)
        m = ((ax - ox) / k, (ay - oy) / k, (bx - ox) / k, (by - oy) / k)

        def box(left: float, top: float, right: float, bottom: float) -> tuple[float, float, float, float]:
            xs, ys = [], []
            for x, y in ((left, top), (right, bottom)):
                xs.append(ox + m[0] * x + m[2] * y)
                ys.append(oy + m[1] * x + m[3] * y)
            return min(xs), min(ys), max(xs), max(ys)

        left, right, bottom, top = (ctypes.c_double() for _ in range(4))
        chars = []
        for i in range(pdfium_c.FPDFText_CountChars(raw)):
            code = pdfium_c.FPDFText_GetUnicode(raw, i)
            ch = chr(code) if code else ""
            if ch == "\ufffe":
                ch = "-"
            angle = pdfium_c.FPDFText_GetCharAngle(raw, i)
            size = pdfium_c.FPDFText_GetFontSize(raw, i)
            bx = None
            if ch.strip():
                pdfium_c.FPDFText_GetCharBox(raw, i, ctypes.byref(left), ctypes.byref(right), ctypes.byref(bottom),
                                             ctypes.byref(top))
                bx = box(left.value, top.value, right.value, bottom.value)
                if bx[2] < 0 or bx[0] > width or bx[3] < 0 or bx[1] > height:
                    continue  # outside the page: nobody sees it
            chars.append((ch, math.degrees(angle) if angle >= 0 else 0.0, bx, size))
    finally:
        textpage.close()
    visible = [c for c in chars if c[2] is not None]
    if not visible:
        return []
    direction = Counter(round(c[1]) % 360 for c in visible).most_common(1)[0][0]
    groups, current = [], []
    for c in chars:
        if c[0] in "\r\n":
            if current:
                groups.append(current)
            current = []
            continue
        if current and c[2] is not None:
            prev = next((p for p in reversed(current) if p[2] is not None), None)
            if prev is not None:
                turned, was_turned = _angle_delta(c[1], direction) > 2, _angle_delta(prev[1], direction) > 2
                if turned != was_turned or (not turned and c[2][0] - prev[2][2] > 2.5 * max(prev[3], c[3], 4)):
                    groups.append(current)
                    current = []
        current.append(c)
    if current:
        groups.append(current)
    lines = []
    for group in groups:
        text = "".join(c[0] for c in group)
        boxes = [c[2] for c in group]
        shown = [c for c in group if c[2] is not None]
        if not shown or not text.strip():
            continue
        turned = sum(_angle_delta(c[1], direction) > 2 for c in shown) > len(shown) / 2
        sizes = [c[3] for c in shown if c[3] > 0]
        size = statistics.median(sizes) if sizes else 0.0
        bottoms = [c[2][3] for c in shown if c[0].isalnum()]
        line = _Line(text, excluded=bool(turned or (size and size < TINY) or _ESIGN.search(text) or _CODE.search(text)),
                     bbox=(min(c[2][0] for c in shown), min(c[2][1] for c in shown),
                           max(c[2][2] for c in shown), max(c[2][3] for c in shown)),
                     baseline=statistics.median(bottoms) if bottoms else max(c[2][3] for c in shown),
                     size=size, boxes=boxes)
        lines.append(line)
    return lines


# ── the check ────────────────────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Lost:
    line: int                 # index in the page's PDFium lines
    tokens: list[int]         # the line's lost words (all of them for a whole line)
    whole: bool


def _lost(lines: list[_Line], read: _Read, short: bool = False) -> list[_Lost]:
    """The PDFium words a read lost: runs of MIN_RUN or more consecutive uncovered words it lacks, and whole lines
    of MIN_RUN words or more it lacks though their three-word runs occur elsewhere on the page. With `short`, also
    whole lines of two to four words (a table's cells): they count for `missing`, never for a recovery."""
    reference = Counter(t for ln in lines for t in ln.tokens)
    body = "".join(t for ln in lines if not ln.excluded for t in ln.tokens)
    seq = []
    for li, ln in enumerate(lines):
        if ln.excluded:
            continue
        for ti, ok in enumerate(read.covered(ln.tokens)):
            seq.append((li, ti, not ok))
    lost: dict[int, set] = {}
    i = 0
    while i < len(seq):
        if not seq[i][2]:
            i += 1
            continue
        j = i
        while j < len(seq) and seq[j][2]:
            j += 1
        group, i = seq[i:j], j
        if len(group) < MIN_RUN:
            continue
        parts: dict[int, list[str]] = {}
        for li, ti, _ in group:
            parts.setdefault(li, []).append(lines[li].tokens[ti])
        if read.lacks(list(parts.values()), reference):
            for li, ti, _ in group:
                lost.setdefault(li, set()).add(ti)
    smallest = 2 if short else MIN_RUN
    for li, ln in enumerate(lines):
        if ln.excluded or li in lost or len(ln.tokens) < smallest or len("".join(ln.tokens)) < 6:
            continue
        if read.lacks([ln.tokens], reference) and not _kept(ln, read, body):
            lost[li] = set(range(len(ln.tokens)))
    out = []
    for li in sorted(lost):
        ln = lines[li]
        whole = len(lost[li]) == len(ln.tokens) or (read.lacks([ln.tokens], reference) and not _kept(ln, read, body))
        out.append(_Lost(li, list(range(len(ln.tokens))) if whole else sorted(lost[li]), whole))
    return out


def _kept(line: _Line, read: _Read, body: str) -> bool:
    """Whether the read holds part of the line as it is: a stretch of its covered words, five or more and twenty
    letters, side by side in the read as often as on PDFium's page (`body`, its words joined), so this occurrence is
    not the one missing. Then the line is not lost whole, only its uncovered words are; a line whose words merely
    recur elsewhere on the page has no such stretch."""
    stretch: list[str] = []
    for token, ok in [*zip(line.tokens, read.covered(line.tokens)), ("", False)]:
        if ok:
            stretch.append(token)
            continue
        joined = "".join(stretch)
        if len(stretch) >= MIN_RUN and len(joined) >= 20 and \
                max(stream.count(joined) for stream in read.joined) >= body.count(joined):
            return True
        stretch = []
    return False


def _body_words(lines: list[_Line]) -> int:
    return sum(len(ln.tokens) for ln in lines if not ln.excluded)


def missing(lines: list[_Line], page: dict) -> int:
    """The criterion between two reads of a page: how many of PDFium's words (outside the lines left out) the read
    lacks, in lost runs of MIN_RUN words or more and in whole lost lines of two words or more. Isolated words a read
    splits or joins differently are not counted."""
    return sum(len(x.tokens) for x in _lost(lines, _Read(page), short=True))


_PROSE = ("paragraph", "heading", "list_item", "caption", "text", "code")


def worse_structure(page: dict, first: dict, text_format: Callable[[dict], str],
                    swallowing: Callable[[dict, float, float], bool]) -> Optional[str]:
    """Why a read's structure is worse than the first read's, or None. Worse is:

    - "prose_in_tables": a fifth or more of the words the first read has in its paragraphs, headings and list items
      sit in this read's tables (words the first read lacked are not counted);
    - "one_table": a table that swallows the page (a few rows, half its area, the body in a cell: the fork's own
      test, `swallowing`) where the first read had none;
    - "fragmented": its prose in more than twice as many blocks as the first read's, and five more at least (an OCR
      read cut into lines);
    - "spatial": the fork would render it as spatial text where it renders the first read's blocks.
    """
    def tables(p: dict) -> Counter:
        return Counter(t for b in p.get("blocks") or [] if b.get("kind") == "table" for t in _tokens(_block_text(b)))

    def prose(p: dict) -> list[dict]:
        return [b for b in p.get("blocks") or [] if b.get("kind") in _PROSE and _block_text(b).strip()]

    first_prose = Counter(t for b in prose(first) for t in _tokens(_block_text(b)))
    gained = tables(page) - tables(first)
    if first_prose and sum(min(n, gained[w]) for w, n in first_prose.items()) >= 0.2 * sum(first_prose.values()):
        return "prose_in_tables"
    if any(swallowing(b, page["width"], page["height"]) for b in page.get("blocks") or []) and \
            not any(swallowing(b, first["width"], first["height"]) for b in first.get("blocks") or []):
        return "one_table"
    before, after = len(prose(first)), len(prose(page))
    if after > 2 * before and after >= before + 5:
        return "fragmented"
    if text_format(page) == "spatial" and text_format(first) == "blocks":
        return "spatial"
    return None


# ── where a lost run goes ────────────────────────────────────────────────────────────────────────────────────


@dataclass
class _Piece:
    """Consecutive lost words of a page (one or more PDFium lines), what surrounds them and where they go."""
    parts: list[_Lost]
    text: str                       # the run as a block's text: its lines joined as LiteParse joins a paragraph's
    rows: list[str]                 # its lines, as a page's spatial text shows them
    before: list[str]               # PDFium's words before the run and after it: the anchors' words
    after: list[str]
    first: _Line
    last: _Line
    previous: Optional[_Line]       # the lines holding the words before and after
    following: Optional[_Line]
    bbox: tuple[float, float, float, float]
    items: list[dict]
    words: int
    place: Optional[str] = None     # inside, append, prepend or new
    x: Optional[int] = None         # the block before it, or holding it
    y: Optional[int] = None         # the block after it
    reason: Optional[str] = None    # why it has no place

    @property
    def hyphen_before(self) -> bool:
        """The run continues a word a line-end hyphen broke («propor-» then «cionado»)."""
        return bool(self.previous is not None and self.previous is not self.first and self.text[:1].islower()
                    and _HYPHEN_END.search(self.previous.text.rstrip()))

    def tokens(self, lines: list[_Line]) -> list[list[str]]:
        return [[lines[p.line].tokens[t] for t in p.tokens] for p in self.parts]


def _part(line: _Line, part: _Lost) -> tuple[str, list]:
    """A lost part's text with its whitespace collapsed, and the box of each of its characters (None for a space)."""
    text = line.text
    if part.whole:
        a, b = 0, len(text)
    else:
        a, b = line.spans[part.tokens[0]][1], line.spans[part.tokens[-1]][2]
        while a > 0 and not text[a - 1].isspace():
            a -= 1
        while b < len(text) and not text[b].isspace():
            b += 1
    out, boxes = [], []
    for k in range(a, b):
        ch = text[k]
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                boxes.append(None)
        else:
            out.append(ch)
            boxes.append(line.boxes[k] if k < len(line.boxes) else None)
    while out and out[-1] == " ":
        out.pop()
        boxes.pop()
    return "".join(out), boxes


def _join(texts: list[str]) -> str:
    """Lines joined as LiteParse joins a paragraph's: a space, or nothing for a word a line-end hyphen broke."""
    out = ""
    for t in texts:
        if not out:
            out = t
        elif _HYPHEN_END.search(out) and t[:1].islower():
            out = out[:-1] + t
        else:
            out += " " + t
    return out


def _union(boxes) -> Optional[tuple[float, float, float, float]]:
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes))


def _item(text: str, boxes: list, rotation: float) -> Optional[dict]:
    """A text item for a recovered line, as LiteParse gives one, with no font metadata: those fields stay empty, as
    LiteParse leaves them when it does not read them. Its word boxes are PDFium's characters' boxes."""
    whole = _union(boxes)
    if whole is None:
        return None
    words, word, word_boxes = [], "", []
    for ch, bx in list(zip(text, boxes)) + [(" ", None)]:
        if ch == " ":
            box = _union(word_boxes)
            if word and box:
                words.append({"text": word, "x": box[0], "y": box[1], "width": box[2] - box[0],
                              "height": box[3] - box[1]})
            word, word_boxes = "", []
        else:
            word += ch
            word_boxes.append(bx)
    x0, y0, x1, y1 = whole
    return {"text": text, "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0,
            "font_name": None, "font_size": None, "font_height": None, "font_ascent": None, "font_descent": None,
            "font_weight": None, "text_width": None, "font_is_buggy": False, "mcid": None, "fill_color": None,
            "stroke_color": None, "char_codes": [], "trailing_space_generated": False, "confidence": None,
            "rotation": rotation, "words": words, "recovered": True}


def _pieces(lines: list[_Line], lost: list[_Lost], rotation: float) -> list[_Piece]:
    """The lost words in runs of consecutive ones, in PDFium's order (the lines left out skipped)."""
    by_line = {x.line: x for x in lost}
    lost_at = {(x.line, t) for x in lost for t in x.tokens}
    seq = [(li, ti) for li, ln in enumerate(lines) if not ln.excluded for ti in range(len(ln.tokens))]
    flags = [key in lost_at for key in seq]
    pieces, i = [], 0
    while i < len(seq):
        if not flags[i]:
            i += 1
            continue
        j = i
        while j < len(seq) and flags[j]:
            j += 1
        line_ids = list(dict.fromkeys(li for li, _ in seq[i:j]))
        parts = []
        for li in line_ids:
            mine = [ti for l2, ti in seq[i:j] if l2 == li]
            parts.append(_Lost(li, mine, by_line[li].whole and len(mine) == len(lines[li].tokens)))
        texts, items = [], []
        for part in parts:
            text, boxes = _part(lines[part.line], part)
            texts.append(text)
            item = _item(text, boxes, rotation)
            if item:
                items.append(item)
        bbox = _union([(t["x"], t["y"], t["x"] + t["width"], t["y"] + t["height"]) for t in items])
        pieces.append(_Piece(
            parts=parts, text=_join(texts), rows=texts,
            before=[lines[li].tokens[ti] for li, ti in seq[max(0, i - 10):i]],
            after=[lines[li].tokens[ti] for li, ti in seq[j:j + 10]],
            first=lines[parts[0].line], last=lines[parts[-1].line],
            previous=lines[seq[i - 1][0]] if i else None, following=lines[seq[j][0]] if j < len(seq) else None,
            bbox=bbox or lines[parts[0].line].bbox or (0.0, 0.0, 0.0, 0.0), items=items, words=j - i))
        i = j
    return pieces


# ── the anchors and the edits ────────────────────────────────────────────────────────────────────────────────


def _anchor(words: list[str], streams: list[str], from_end: bool) -> Optional[tuple[int, int, int]]:
    """Where the words next to a lost run occur once in the streams (letters alone): (stream, start, end), in
    letters. The fewest of them, two to eight and eight letters or more, that occur once."""
    if not words:
        return None
    for k in range(2, min(8, len(words)) + 1):
        chosen = words[-k:] if from_end else words[:k]
        key = "".join(_letter(c) for c in "".join(chosen))
        if not key or (len(key) < 8 and k < min(8, len(words))):
            continue
        hits = [(s, m.start()) for s, stream in enumerate(streams) for m in re.finditer(re.escape(key), stream)]
        if len(hits) == 1:
            s, start = hits[0]
            return s, start, start + len(key)
        if not hits:
            return None
    return None


@dataclass
class _Edit:
    start: int
    end: int
    text: str


def _apply(text: str, edits: list[_Edit]) -> tuple[str, list[list]]:
    """Edits of a text, none overlapping: the new text and its recovered spans, [start, end, what they replaced]."""
    out, spans, cursor, shift = [], [], 0, 0
    for e in sorted(edits, key=lambda e: (e.start, e.end)):
        if e.start < cursor:
            raise ValueError("overlapping recovery edits")
        out.append(text[cursor:e.start])
        start = e.start + shift
        spans.append([start, start + len(e.text), text[e.start:e.end]])
        out.append(e.text)
        shift += len(e.text) - (e.end - e.start)
        cursor = e.end
    out.append(text[cursor:])
    return "".join(out), spans


def _edit(text: str, stream: tuple[str, list[int]], piece: _Piece, style: str, *,
          before: bool = True, after: bool = True) -> Optional[_Edit]:
    """The edit that puts a run into a text (a block's, the page's markdown) between its anchors: `inside` a
    paragraph, appended to the end of the one before (`append`), at the start of the one after (`prepend`), or a
    paragraph of its own (`new`). None when the anchors are not there once and side by side."""
    letters, where = stream
    prev = _anchor(piece.before, [letters], from_end=True) if before and piece.before else None
    nxt = _anchor(piece.after, [letters], from_end=False) if after and piece.after else None
    if (before and piece.before and prev is None) or (after and piece.after and nxt is None):
        return None
    if prev and nxt and prev[2] != nxt[1]:
        return None
    k = where[prev[2] - 1] + 1 if prev else None     # just after the last letter before the run
    s = where[nxt[1]] if nxt else None               # the first letter after it
    stop = s if s is not None else len(text)
    body = piece.text
    joined_after = bool(s is not None and _HYPHEN_END.search(body) and text[s].islower())
    if style in ("inside", "append", "new"):
        if k is None:
            if style != "new":
                return None
        else:
            i = k
            while i < stop and not text[i].isspace():
                i += 1
    if style in ("inside", "prepend") or (style == "new" and k is None):
        if s is None:
            return None
        j = s
        while j > (k or 0) and not text[j - 1].isspace():
            j -= 1
    if style == "inside":
        if "\n\n" in text[i:j]:
            return None
        lead = " "
        if piece.hyphen_before:
            i, lead = k, ""
            if text[k:k + 1] == "-":
                pass  # the hyphen LiteParse kept goes: text[k:j] is replaced
        if joined_after:
            return _Edit(i, s, lead + body[:-1])
        return _Edit(i, j, lead + body + " ")
    if style == "append":
        if piece.hyphen_before:
            return _Edit(k, k + 1 if text[k:k + 1] == "-" else k, body)
        return _Edit(i, i, " " + body)
    if style == "prepend":
        if joined_after:
            return _Edit(j, j, body[:-1])
        return _Edit(j, j, body + " ")
    if k is None:
        line_start = text.rfind("\n", 0, s) + 1
        return _Edit(line_start, line_start, body + "\n\n")
    return _Edit(i, i, "\n\n" + body)


def _rows(text: str, stream: tuple[str, list[int]], piece: _Piece, lines: list[_Line]) -> Optional[_Edit]:
    """The edit that puts a run into the page's spatial text: its lines as rows of their own between the rows of
    its anchors, in the blank rows the read left there when there are enough (as far down as PDFium places them),
    indented as the row before when they start where it starts. A part of a line goes inside its row."""
    letters, where = stream
    prev = _anchor(piece.before, [letters], from_end=True) if piece.before else None
    nxt = _anchor(piece.after, [letters], from_end=False) if piece.after else None
    if (piece.before and prev is None) or (piece.after and nxt is None):
        return None
    if prev and nxt and prev[2] != nxt[1]:
        return None
    if not all(p.whole for p in piece.parts):
        edit = _edit(text, stream, piece, "inside")
        return None if edit is None or "\n" in text[edit.start:edit.end] else edit
    i = j = None
    if prev:
        nl = text.find("\n", where[prev[2] - 1] + 1)
        i = len(text) if nl < 0 else nl            # the end of the row holding the words before
    if nxt:
        j = text.rfind("\n", 0, where[nxt[1]])      # the line break before the row holding the words after

    def indent(line: Optional[_Line], at: int) -> Optional[str]:
        if line is None or not line.bbox or not piece.first.bbox or abs(line.bbox[0] - piece.first.bbox[0]) > 3:
            return None
        row = text[text.rfind("\n", 0, at) + 1:]
        row = row.split("\n", 1)[0]
        return row[:len(row) - len(row.lstrip())]

    lead = indent(piece.previous, i) if i is not None else None
    if lead is None and j is not None:
        lead = indent(piece.following, j + 1)
    rows = [(lead or "") + r for r in piece.rows]
    if i is None:
        start = j + 1 if j is not None and j >= 0 else 0
        return _Edit(start, start, "\n".join(rows) + "\n")
    if j is None:
        return _Edit(i, i, "".join("\n" + r for r in rows))
    if j < i:
        return None
    blanks = text[i:j].split("\n")[1:]
    if any(b.strip() for b in blanks):
        return None
    skip = 0
    if piece.previous and piece.previous.baseline is not None and piece.first.baseline is not None:
        pitch = _pitch(lines, piece.first.size or 10)
        skip = max(0, round((piece.first.baseline - piece.previous.baseline) / pitch) - 1)
    if len(blanks) >= len(rows):
        skip = min(skip, len(blanks) - len(rows))
        kept = blanks[:skip] + rows + blanks[skip + len(rows):]
    else:
        skip = min(skip, len(blanks))
        kept = blanks[:skip] + rows + blanks[skip:]
    return _Edit(i, j, "".join("\n" + b for b in kept))


def _pitch(lines: list[_Line], size: float) -> float:
    """The page's line pitch for type of this size: the median step between consecutive baselines."""
    steps = []
    body = [ln for ln in lines if not ln.excluded and ln.baseline is not None]
    for a, b in zip(body, body[1:]):
        step = b.baseline - a.baseline
        if 0.8 * size <= step <= 2.2 * size:
            steps.append(step)
    return statistics.median(steps) if steps else 1.2 * size


def _grow(bbox: Optional[dict], boxes: list) -> Optional[dict]:
    boxes = [b for b in boxes if b]
    if not bbox or not boxes:
        return bbox
    x0, y0 = min([bbox["x"]] + [b[0] for b in boxes]), min([bbox["y"]] + [b[1] for b in boxes])
    x1 = max([bbox["x"] + bbox["width"]] + [b[2] for b in boxes])
    y1 = max([bbox["y"] + bbox["height"]] + [b[3] for b in boxes])
    return {**bbox, "x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0}


def _new_block(piece: _Piece) -> dict:
    """A paragraph block of LiteParse's for a run between two blocks, its whole text recovered."""
    x0, y0, x1, y1 = piece.bbox
    return {"kind": "paragraph", "text": piece.text, "level": None, "bold": False, "italic": False, "ordered": None,
            "marker": None, "lines": None, "lang": None, "header": None, "rows": None, "id": None, "format": None,
            "bbox": {"x": x0, "y": y0, "width": x1 - x0, "height": y1 - y0},
            "recovered": [[0, len(piece.text), ""]]}


_INSIDE = ("paragraph", "list_item", "heading", "caption", "text")


def _place(piece: _Piece, blocks: list[dict], streams: list[str], lines: list[_Line]) -> None:
    """Where a run goes in the read's blocks, from its anchors: inside the block holding both side by side, or
    between two consecutive blocks, the one ending with the words before and the one starting with the words after:
    appended to the first when no paragraph gap separates it from its line before, at the start of the second when
    none separates it from its line after (a paragraph block only), else a paragraph of its own."""
    piece.place, piece.x, piece.y, piece.reason = None, None, None, None
    prev = _anchor(piece.before, streams, from_end=True) if piece.before else None
    nxt = _anchor(piece.after, streams, from_end=False) if piece.after else None
    if (piece.before and prev is None) or (piece.after and nxt is None) or (prev is None and nxt is None):
        piece.reason = "anchor"
        return
    kinds = [b.get("kind") for b in blocks]
    if (prev and kinds[prev[0]] == "table") or (nxt and kinds[nxt[0]] == "table"):
        piece.reason = "table"
        return
    if prev and nxt and prev[0] == nxt[0]:
        if prev[2] != nxt[1]:
            piece.reason = "apart"
        elif kinds[prev[0]] not in _INSIDE:
            piece.reason = "block_kind"
        else:
            piece.place, piece.x = "inside", prev[0]
        return
    if (prev and prev[2] != len(streams[prev[0]])) or (nxt and nxt[1] != 0) or \
            (prev and nxt and (nxt[0] < prev[0] or any(streams[k] for k in range(prev[0] + 1, nxt[0])))):
        piece.reason = "apart"
        return
    x, y = (prev[0] if prev else None), (nxt[0] if nxt else None)
    pitch = _pitch(lines, piece.first.size or 10)

    def near(a: Optional[_Line], b: Optional[_Line]) -> bool:
        return bool(a is not None and b is not None and a.baseline is not None and b.baseline is not None
                    and 0 < b.baseline - a.baseline <= 1.3 * pitch)

    near_before, near_after = near(piece.previous, piece.first), near(piece.last, piece.following)
    can_append = x is not None and kinds[x] in ("paragraph", "list_item")
    can_prepend = y is not None and kinds[y] == "paragraph"
    if near_before and can_append:
        piece.place = "append"
    elif near_after and can_prepend:
        piece.place = "prepend"
    else:
        piece.place = "new"
    piece.x, piece.y = x, y


def _recover(page: dict, lines: list[_Line], pieces: list[_Piece]) -> Optional[dict]:
    """The page with every run that has a place in its blocks, its markdown and its text; None when none has."""
    blocks = page.get("blocks") or []
    block_texts = [(b.get("text") or "") if b.get("kind") != "table" else _block_text(b) for b in blocks]
    block_streams = [_letters(t) for t in block_texts]
    streams = [s for s, _ in block_streams]
    markdown, text = page.get("markdown") or "", page.get("text") or ""
    md_stream, text_stream = _letters(markdown), _letters(text)
    items = page.get("text_items") or []
    item_streams = [_letters(t.get("text") or "")[0] for t in items]
    block_edits: dict[int, list[tuple[_Edit, _Piece]]] = {}
    new_blocks: list[_Piece] = []
    md_edits, text_edits, inserts = [], [], []
    for piece in pieces:
        _place(piece, blocks, streams, lines)
        if piece.place is None:
            continue
        md = _edit(markdown, md_stream, piece, piece.place)
        tx = _rows(text, text_stream, piece, lines)
        target, block_edit = None, None
        if piece.place == "inside":
            target = piece.x
            block_edit = _edit(block_texts[target], block_streams[target], piece, "inside")
        elif piece.place == "append":
            target = piece.x
            block_edit = _edit(block_texts[target], block_streams[target], piece, "append", after=False)
        elif piece.place == "prepend":
            target = piece.y
            block_edit = _edit(block_texts[target], block_streams[target], piece, "prepend", before=False)
        if md is None or tx is None or (target is not None and block_edit is None):
            piece.reason = "markdown" if md is None else "text" if tx is None else "block"
            piece.place = None
            continue
        if target is None:
            new_blocks.append(piece)
        else:
            block_edits.setdefault(target, []).append((block_edit, piece))
        md_edits.append(md)
        text_edits.append(tx)
        nxt = _anchor(piece.after, item_streams, from_end=False) if piece.after else None
        prev = _anchor(piece.before, item_streams, from_end=True) if piece.before else None
        if nxt is not None:
            at = nxt[0] if nxt[1] == 0 else nxt[0] + 1
        elif prev is not None:
            at = prev[0] + 1
        else:
            at = next((n for n, t in enumerate(items) if t.get("y", 0) > piece.bbox[1]), len(items))
        inserts.append((at, piece))
    placed = [p for p in pieces if p.place]
    if not placed:
        return None
    out = dict(page)
    new_list = []
    for k, block in enumerate(blocks):
        new_list.extend(_new_block(p) for p in new_blocks if p.x is None and p.y == k)
        if k in block_edits:
            block = dict(block)
            edits = [e for e, _ in block_edits[k]]
            block["text"], spans = _apply(block.get("text") or "", edits)
            block["bbox"] = _grow(block.get("bbox"), [p.bbox for _, p in block_edits[k]])
            block["recovered"] = _moved(block.get("recovered") or [], edits) + spans
        new_list.append(block)
        new_list.extend(_new_block(p) for p in new_blocks if p.x == k)
    out["blocks"] = new_list
    out["markdown"], md_spans = _apply(markdown, md_edits)
    out["text"], text_spans = _apply(text, text_edits)
    old = page.get("recovered") or {}
    out["recovered"] = {"text": _moved(old.get("text") or [], text_edits) + text_spans,
                        "markdown": _moved(old.get("markdown") or [], md_edits) + md_spans}
    new_items = list(items)
    for at, piece in sorted(inserts, key=lambda x: x[0], reverse=True):
        new_items[at:at] = piece.items
    out["text_items"] = new_items
    return out


def _moved(spans: list[list], edits: list[_Edit]) -> list[list]:
    """Recovered spans of a text, moved where these later edits leave them."""
    out = []
    for s, e, was in spans:
        delta = sum(len(x.text) - (x.end - x.start) for x in edits if x.end <= s)
        out.append([s + delta, e + delta, was])
    return sorted(out)


def restore(text: str, spans: list[list]) -> str:
    """A text without its recovered spans: as the read gave it."""
    for s, e, was in sorted(spans, reverse=True):
        text = text[:s] + was + text[e:]
    return text


def _unrecovered(blocks: list[dict]) -> list[dict]:
    """Blocks without their recovered spans; a block of recovered text alone dropped."""
    kept = []
    for b in blocks:
        spans = b.get("recovered")
        if spans:
            b = {key: value for key, value in b.items() if key != "recovered"}
            b["text"] = restore(b.get("text") or "", spans)
            if not b["text"].strip():
                continue
        kept.append(b)
    return kept


def as_read(record: dict) -> dict:
    """A stored page as LiteParse's read gave it, before the recovery: its blocks' recovered spans undone (a block
    of recovered text alone dropped), its markdown joined again from them, and in its layout the recovered text
    items left out and LiteParse's blocks, text and markdown as they were. The boxes of the blocks that took
    recovered text stay as they grew. A page with nothing recovered is returned as it is."""
    blocks = record.get("blocks") or []
    layout = record.get("layout") or {}
    items = layout.get("text_items") or []
    if not any(b.get("recovered") for b in blocks) and not any(t.get("recovered") for t in items) \
            and not layout.get("recovered"):
        return record
    kept = _unrecovered(blocks)
    out = dict(record)
    out["blocks"] = kept
    out["markdown"] = "\n\n".join(b["text"] for b in kept)
    out["citation_markdown"] = "\n\n".join(f"[block_id={b['block_id']}]\n{b['text']}" for b in kept)
    read = {key: value for key, value in layout.items() if key != "recovered"}
    read["text_items"] = [t for t in items if not t.get("recovered")]
    if layout.get("blocks"):
        read["blocks"] = _unrecovered(layout["blocks"])
    spans = layout.get("recovered") or {}
    for key in ("text", "markdown"):
        if spans.get(key):
            read[key] = restore(layout.get(key) or "", spans[key])
    out["layout"] = read
    return out


# ── the read checked, page by page ───────────────────────────────────────────────────────────────────────────


def _rotation(page: dict) -> float:
    """The direction LiteParse gives most of a page's text, for the items made from PDFium's lines."""
    turns = Counter(float(t.get("rotation") or 0.0) for t in page.get("text_items") or []
                    if (t.get("text") or "").strip())
    return turns.most_common(1)[0][0] if turns else 0.0


def _attempt(how: str, page: dict, lines: list[_Line], reference: Counter, lost: Optional[list[_Lost]] = None) -> dict:
    """One read of a page checked and recovered: the page as it ends, its runs and the words it still lacks."""
    if lost is None:
        lost = _lost(lines, _Read(page))
    pieces = _pieces(lines, lost, _rotation(page))
    recovered = _recover(page, lines, pieces) if pieces else None
    if recovered is not None:
        # a run the recovered page still lacks did not go in as it should: taken out again
        check = _Read(recovered)
        bad = [p for p in pieces if p.place and check.lacks(p.tokens(lines), reference)]
        if bad:
            good = [p for p in pieces if p.place and p not in bad]
            for p in pieces:
                p.place, p.x, p.y = None, None, None
            recovered = _recover(page, lines, good) if good else None
            for p in bad:
                p.reason = "check"
    result = recovered if recovered is not None else page
    return {"read": how, "page": result, "pieces": pieces, "lost": lost, "missing": missing(lines, result),
            "placed": [p for p in pieces if p.place], "unplaced": [p for p in pieces if not p.place]}


def _check_page(number: int, page: dict, pdf, reread: Callable[[int, str], dict], ocr: bool,
                text_format: Callable[[dict], str],
                swallowing: Callable[[dict, float, float], bool]) -> tuple[Optional[dict], Optional[dict]]:
    """One native page: (the page to keep, or None when it stays as read; its report entry, or None when nothing
    was lost)."""
    started = perf_counter()
    with PDFIUM_LOCK:
        pdf_page = pdf[number - 1]
        try:
            textpage = pdf_page.get_textpage()
            try:
                quick = textpage.get_text_range()
            finally:
                textpage.close()
        except BaseException:
            pdf_page.close()
            raise
    read = _Read(page)
    try:
        if not _lost(_quick_lines(quick), read):
            return None, None
        with PDFIUM_LOCK:
            lines = _geometry_lines(pdf_page)
    finally:
        with PDFIUM_LOCK:
            pdf_page.close()
    lost = _lost(lines, read)
    if not lost:
        return None, None
    reference = Counter(t for ln in lines for t in ln.tokens)
    entry: dict[str, Any] = {"page": number, "words": _body_words(lines)}
    # Text a clip path hides is not lost: LiteParse leaves it out on purpose. Its lines are left out from here on.
    shown = _Read(reread(number, "clipped"))
    hidden = {x.line for x in lost if not shown.lacks([[lines[x.line].tokens[t] for t in x.tokens]], reference)}
    if hidden:
        entry["clipped_words"] = sum(len(x.tokens) for x in lost if x.line in hidden)
        lines = [_Line(ln.text, excluded=True, bbox=ln.bbox, baseline=ln.baseline, size=ln.size, boxes=ln.boxes)
                 if k in hidden else ln for k, ln in enumerate(lines)]
        lost = [x for x in lost if x.line not in hidden]
    if not lost:
        entry.update(action="none", seconds=round(perf_counter() - started, 3))
        return None, entry
    first = _attempt("liteparse", page, lines, reference, lost)
    options, tried = [first], {}
    if first["unplaced"]:
        for how in ("text_metadata", "ocr"):
            if how == "ocr" and not ocr:
                break
            try:
                other = _attempt(how, reread(number, how), lines, reference)
            except Exception as exc:  # the first read stands
                tried[how] = {"error": f"{type(exc).__name__}: {exc}"[:200]}
                continue
            other["worse"] = worse_structure(other["page"], page, text_format, swallowing)
            tried[how] = {"missing": other["missing"], **({"placed": len(other["placed"])} if other["placed"] else {}),
                          **({"worse": other["worse"]} if other["worse"] else {})}
            options.append(other)
            if not other["unplaced"] and not other["worse"]:
                break
    best = first
    for other in options[1:]:
        if other.get("worse") is None and other["missing"] < best["missing"]:
            best = other
    if best is first:
        action = "recovered" if first["placed"] else "unrecovered"
    else:
        action = "reread"
    entry.update({
        "action": action, "read": best["read"],
        "lost": {"lines": len(lost), "words": sum(len(x.tokens) for x in lost)},
        # what the recovery did on the first read; a read kept instead says so in `read`, its own in `tried`
        "placed": dict(Counter(p.place for p in first["placed"])),
        "unplaced": dict(Counter(p.reason for p in first["unplaced"])),
        "missing": {"before": missing(lines, page), "after": best["missing"]},
        "runs": [[round(p.bbox[1], 1), p.words, p.place or p.reason] for p in first["pieces"]],
    })
    if tried:
        entry["tried"] = tried
    entry["seconds"] = round(perf_counter() - started, 3)
    if best is first and not first["placed"]:
        return None, entry
    return best["page"], entry


def check_pages(data: bytes, pages: dict[int, dict], native: list[int], reread: Callable[[int, str], dict], *,
                ocr: bool, text_format: Callable[[dict], str],
                swallowing: Callable[[dict, float, float], bool]) -> tuple[dict, dict[int, dict]]:
    """Check the native pages of a read (LiteParse's pages by number) against PDFium's text layer and put back what
    they lost (module docstring). `reread(number, how)` reads one page again: "clipped" (clipped text kept),
    "text_metadata" or "ocr" (OCR with the text metadata); OCR only when `ocr`. `text_format(page)` is the
    fork's choice of blocks or spatial text for a page, `swallowing(block, width, height)` its test of a table that
    swallows the page.

    Returns the report (pages checked, one entry per page where something was lost, seconds) and, per page that
    changes, its entry; `pages` holds the changed pages in place of the read's."""
    started = perf_counter()
    report: dict[str, Any] = {"checked": 0, "pages": []}
    changed: dict[int, dict] = {}
    try:
        import pypdfium2 as pdfium
        with PDFIUM_LOCK:
            pdf = pdfium.PdfDocument(data)
    except Exception as exc:  # the read stands as it is
        report["error"] = f"{type(exc).__name__}: {exc}"[:200]
        report["seconds"] = round(perf_counter() - started, 3)
        return report, changed
    try:
        for number in native:
            page = pages[number]
            report["checked"] += 1
            new, entry = _check_page(number, page, pdf, reread, ocr, text_format, swallowing)
            if entry is not None:
                report["pages"].append(entry)
            if new is not None:
                pages[number] = new
                changed[number] = entry
    finally:
        with PDFIUM_LOCK:
            pdf.close()
    report["seconds"] = round(perf_counter() - started, 3)
    return report, changed


def document_text(text: str, before: list[dict], after: list[dict]) -> str:
    """LiteParse's text of the whole document again, after some pages changed: joined as it was joined (its pages'
    markdown with LiteParse's page rule, or their text with blank lines, as the fork joins an OCR pass in), or else
    each changed page's markdown replaced where it occurs."""
    for key, joiner in (("markdown", "\n\n-----\n\n"), ("text", "\n\n")):
        if text == joiner.join(str(p.get(key) or "") for p in before):
            return joiner.join(str(p.get(key) or "") for p in after)
    for old, new in zip(before, after):
        for key in ("markdown", "text"):
            a, b = str(old.get(key) or ""), str(new.get(key) or "")
            if a != b and a and text.count(a) == 1:
                text = text.replace(a, b)
                break
    return text
