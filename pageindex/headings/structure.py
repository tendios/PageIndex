"""Mode headings, second reading: every line that could open a section (blocks, table rows, titles that run into
the text of a block), the document's index pages found by their leaders or by titles that come back later in
order, and a numbering spine that follows each family of markers (decimal, letter, roman, ordinal, keyword) the
way a reader does: a line continues the sequence (next sibling, first child) or it does not, and a restart needs
the look of a heading and a continuation after it. What the spine and the style cannot settle goes to the judge.
Ranks come from the families («Capítulo» over «Cláusula» or «Primera» over «1.1»), so the tree follows the
document's own hierarchy."""
from __future__ import annotations

import re
from collections import Counter
from typing import Any, Optional

from .candidates import (LETTER_ITEM, MAX_TEXT, Candidate, _carries_page_number, _closest, _flatten,
                         _located_index_entries, _order_in_page, _snippet, is_repeated, masked, norm, repeated_texts,
                         similar, strip_num, tableish)
from .numbering import Marker, first_of_sequence, follows, parse

LOOKAHEAD = 60          # candidates searched for the continuation of a start or a restart
GAP_MAX = 3             # numbers a sequence may skip («B1» then «B3») when the line looks like a heading
CLAUSE_FAMILIES = ("ordinal", "clause")
TOP_FAMILIES = ("chapter", "annex", "roman")
_LEADERS = re.compile(r"(?:\.\s?){4,}|…{2,}|_{5,}")
_LEADER_END = re.compile(r"(?:(?:\.\s?){4,}|…{2,}|_{5,})\s*(?:\d{1,4}|¡?error.*|p[aà]g.*)?\s*$", re.I)
_REFERENCE_BEFORE = re.compile(r"(?:^|[\s(])(?:apartad[oa]s?|apartats?|cl[aá]usulas?|art[ií]culos?|art\.|articles?|punto|punt|"
                               r"ep[ií]grafe|anexo|annex|del|de la|de l'|en el|al|la|el|y|i|o|según|segons|per)\s*$", re.I)
_ORDINAL_WORDS = (r"(?:Primer[ao]?|Segund[ao]|Segona|Tercer[ao]?|Cuart[ao]|Quarta|Quint[ao]|Cinquena|Sext[ao]|Sisena|"
                  r"S[eé]ptim[ao]|Setena|Octav[ao]|Vuitena|Noven[ao]|D[eé]cim[ao]|Desena|Onzena|Dotzena|Tretzena|"
                  r"Catorzena|Quinzena|Setzena|Dissetena|Divuitena|Dinovena|Vintena|Vint-i-\w+|Trentena|Trenta-\w+|"
                  r"Quarantena|Quaranta-\w+|Vig[eé]sim[ao](?:\s\w+)?|Trig[eé]sim[ao](?:\s\w+)?|Und[eé]cim[ao]|"
                  r"Duod[eé]cim[ao]|Decimo\w+|Vigesimo\w+)")
_INLINE = re.compile(
    # after a sentence or a title's end: «…de crèdit. 3.5 S'han…», «…(annex 2). B.1 Pressupost…»
    r"(?<=[.:;)»”\]])\s+(?=(?:\d{1,2}(?:\.\d{1,2}){1,5}(?:\.\s*|\s+)[A-ZÀ-ÚÑÇL]|\d{1,2}\.\s+[A-ZÀ-ÚÑÇ]{2}|"
    r"[A-Z]\.?\d{1,2}\.?\s+[A-ZÀ-ÚÑÇ]|" + _ORDINAL_WORDS + r"\s*[.:]\s+[A-ZÀ-ÚÑÇ]))"
    # or straight after a title with no stop: «…DEL CONTRACTE Dinovena. Condicions…», «…contractista 26.1 L'import…»
    r"|\s+(?=(?:" + _ORDINAL_WORDS + r"\s*\.\s+[A-ZÀ-ÚÑÇ]|\d{1,2}(?:\.\d{1,2}){1,5}\.?\s+[A-ZÀ-ÚÑÇ][a-zà-ÿ'’]|"
    r"(?:Cl[aàá]usula|CL[AÀÁ]USULA|Art[ií]culo|Article)\s+\d{1,3}\s*[.:\-–]\s*[A-ZÀ-ÚÑÇ]))")
_BOLD_MARKS = re.compile(r"(\*{1,3}|_{2,3})(?=\S)|(?<=\S)(\*{1,3}|_{2,3})")


def _clean(text: str) -> tuple[str, bool]:
    bold = "**" in text or "__" in text
    return _BOLD_MARKS.sub("", re.sub(r"^#+\s*", "", text)).strip(), bold


def _font(page: dict[str, Any], block: dict[str, Any]) -> tuple[Optional[float], bool]:
    """The size and weight of the first text item inside a block's box, from LiteParse's layout."""
    box = block.get("bbox_points") or {}
    items = (page.get("layout") or {}).get("text_items") or []
    if not box or not items:
        return None, False
    x0, y0 = box.get("x", 0) - 2, box.get("y", 0) - 2
    x1, y1 = x0 + box.get("width", 0) + 4, y0 + box.get("height", 0) + 4
    inside = [t for t in items if x0 <= t.get("x", -1) <= x1 and y0 <= t.get("y", -1) <= y1 and (t.get("text") or "").strip()]
    if not inside:
        return None, False
    first = min(inside, key=lambda t: (round(t["y"]), t["x"]))
    name = str(first.get("font_name") or "").lower()
    weight = first.get("font_weight") or 0
    return (round(float(first["font_size"]), 1) if first.get("font_size") else None,
            "bold" in name or "black" in name or "heavy" in name or weight >= 600)


def _body_style(pages: list[dict[str, Any]]) -> tuple[Optional[float], bool]:
    """The size most of the running text is set in, and whether that text is bold itself (then bold tells
    nothing): pages kept as spatial text (price lists, tables) excluded, their small print would make every line of
    the body look larger than the body."""
    sizes: Counter = Counter()
    bold = plain = 0
    for page in pages:
        if (page.get("extraction") or {}).get("text_format") == "spatial":
            continue
        for t in (page.get("layout") or {}).get("text_items") or []:
            n = len(t.get("text") or "")
            if t.get("font_size"):
                sizes[round(float(t["font_size"]), 1)] += n
            name = str(t.get("font_name") or "").lower()
            if "bold" in name or "black" in name or (t.get("font_weight") or 0) >= 600:
                bold += n
            else:
                plain += n
    return (sizes.most_common(1)[0][0] if sizes else None), bold > plain


def _body_size(pages: list[dict[str, Any]]) -> Optional[float]:
    return _body_style(pages)[0]


def _candidate(out: list[Candidate], page_no: int, order: int, text: str, *, md: int = 0, heading_block: bool = False,
               bold: bool = False, size: Optional[float] = None, source: str = "liteparse", inline: bool = False,
               previous: str = "", following: str = "", block: Optional[dict] = None, bare: bool = False) -> None:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return
    mark = parse(text, bare=bare or source == "table" or heading_block)
    family = mark.family if mark else None
    out.append(Candidate(index=len(out), page=page_no, order=order, text=text[:400], md=md, heading_block=heading_block,
                         marker=mark.raw if mark else None, numbered=mark is not None,
                         keyword=family if family in ("chapter", "clause", "annex", "ordinal") else None,
                         sources={source}, previous=previous, following=following, bold=bold,
                         page_numbered=_carries_page_number(text, page_no, family if family not in (None, "decimal") else None),
                         leaders=bool(_LEADER_END.search(text)) or bool(_LEADERS.search(text) and re.search(r"\d\s*$", text)),
                         mark=mark, size=size, inline=inline, block_id=(block or {}).get("block_id"),
                         bbox=(block or {}).get("bbox")))


def _title_like(text: str) -> bool:
    """A line of a spatial page that could be an unnumbered title: short, starting with a capital, not ending a
    clause with a comma, and either in capitals or a few words."""
    text = text.strip()
    if not text or len(text) > 100 or not (text[0].isupper() or text[0] in "«\"'¿¡") or text.endswith((",", ";")):
        return False
    letters = [ch for ch in text if ch.isalpha()]
    caps = bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.6
    return caps or (len(text.split()) <= 12 and not re.search(r"[a-zà-ÿ][.;:]\s+[a-zà-ÿ]", text))


def _sequenced(rows: list[Candidate]) -> bool:
    """Whether the numbered rows of a table follow one another (a summary sheet: 1, 2, 3 or A.1, A.2, B) rather
    than repeating (quantities, codes)."""
    paths = [c.mark.path for c in rows if c.mark]
    steps = sum(1 for a, b in zip(paths, paths[1:]) if follows(a, b))
    return steps >= 0.5 * (len(paths) - 1)


def _split_inline(text: str) -> list[str]:
    """A block cut where a numbered title starts inside it: «Novena. Aptitud per contractar 9.1 Estan…» gives
    «Novena. Aptitud per contractar» and «9.1 Estan…». A number that a reference word precedes («apartado 3.1»,
    «art. 22.1») is no title and stays."""
    parts, last = [], 0
    for m in _INLINE.finditer(text):
        before = text[max(0, m.start() - 20):m.start()]
        if _REFERENCE_BEFORE.search(before.rstrip(" .:;)»”]")):
            continue
        parts.append(text[last:m.start()])
        last = m.end()
    parts.append(text[last:])
    return [p.strip() for p in parts if p.strip()]


def lines_of(pages: list[dict[str, Any]]) -> list[Candidate]:
    out: list[Candidate] = []
    body_bold = _body_style(pages)[1]
    for page in pages:
        page_no = int(page.get("page_index") or 0)
        blocks = page.get("blocks") or []
        for k, block in enumerate(blocks):
            kind = block.get("block_type") or ""
            raw = (block.get("text") or "").strip()
            if not raw:
                continue
            size, font_bold = _font(page, block)
            font_bold = font_bold and not body_bold
            previous = _snippet(blocks[k - 1].get("text") if k else "", tail=True)
            following_block = blocks[k + 1].get("text") if k + 1 < len(blocks) else ""
            if kind == "table":
                before = len(out)
                rows = [r for r in raw.split("\n") if r.strip().startswith("|") and not re.fullmatch(r"[|\s:-]+", r.strip())]
                for r, row in enumerate(rows):
                    cells = [c.strip() for c in re.split(r"(?<!\\)\|", row.strip().strip("|"))]
                    cells = [re.sub(r"<br>|\s+", " ", c).replace("\\|", "|").strip() for c in cells if c.strip()]
                    if not cells:
                        continue
                    title, bold = _clean(cells[0])
                    if re.fullmatch(r"\d{1,2}(?:\.\d{1,2}){0,3}\.?|[A-Z](?:\.?\d{1,2}){0,2}\.?", title) and len(cells) > 1:
                        title = f"{title} {_clean(cells[1])[0]}"
                    mark = parse(title, bare=True)
                    if not mark or len(title) > MAX_TEXT:
                        continue
                    _candidate(out, page_no, k * 1000 + r, title, bold=bold, source="table",
                               following=" ".join(cells[1:])[:120], block=block)
                rows_found = out[before:]
                if len(rows_found) >= 3 and not _sequenced(rows_found):
                    del out[before:]  # quantities of an inventory («1 COCINA», «2 BATIDORAS»), not sections
                continue
            lines = [raw.split("\n")[0]] if kind != "page" else [l for l in raw.split("\n") if l.strip() and not l.startswith("```")]
            for sub, line in enumerate(lines):
                md = len(re.match(r"^(#*)", line.strip()).group(1))
                text, bold = _clean(line)
                if kind == "list_item":
                    text = re.sub(r"^[-*•]\s+", "", text)
                pieces = _split_inline(text) if kind != "page" else [text]
                if kind != "page":
                    # Later lines of the block that open with a number («3.1.4. El adjudicatario…») and titles that
                    # run into a line («…de crèdit. 3.5 S'han complert…»): the reader merged them with the block.
                    prev_line = text
                    for l in raw.split("\n")[1:]:
                        line_text = _clean(l)[0]
                        if not line_text:
                            continue
                        opens = parse(line_text) and not _REFERENCE_BEFORE.search(prev_line.rstrip(" .:;)»”]")[-25:])
                        split = _split_inline(line_text)
                        pieces.extend((split if opens else split[1:]))
                        prev_line = line_text
                spatial_row = False
                for n, piece in enumerate(pieces):
                    heading_block = n == 0 and (kind == "heading" or md > 0)
                    if n == 0 and not heading_block and not parse(piece) and not (bold or font_bold) and kind != "page":
                        continue
                    spatial_row = kind == "page" and bool(re.match(r"^\s*\d{1,2}\s+[A-ZÀ-ÚÑÇ]{3}", line))
                    if kind == "page" and not parse(piece, bare=spatial_row) and not _title_like(piece):
                        continue  # a wrapped line of running text on a spatial page: never a title, never asked
                    if n > 0 and not parse(piece):
                        continue
                    _candidate(out, page_no, k * 1000 + sub * 10 + n, piece, md=md if n == 0 else 0,
                               heading_block=heading_block, bold=(bold or font_bold) and n == 0, size=size if n == 0 else None,
                               inline=n > 0, previous=previous if n == 0 else _snippet(pieces[n - 1], tail=True),
                               following=_snippet(pieces[n + 1] if n + 1 < len(pieces) else following_block), block=block,
                               bare=kind == "page" and spatial_row)
    _merge_tagged(out, pages)
    _merge_inspector(out, pages)
    return out


def _merge_inspector(out: list[Candidate], pages: list[dict[str, Any]]) -> None:
    """pdf-inspector's headings: a second opinion on the lines it names (source «inspector»). Alone it decides
    nothing."""
    for page in pages:
        page_no = int(page.get("page_index") or 0)
        on_page = [c for c in out if c.page == page_no]
        for k, head in enumerate(page.get("inspector_headings") or []):
            text = head["text"]
            match = max(((similar(c.text, text), c) for c in on_page), key=lambda x: x[0], default=(0.0, None))
            if match[1] is not None and match[0] >= 0.8:
                match[1].sources.add("inspector")
                continue
            # a heading no line carries is not added: on its own, pdf-inspector names form labels too


def _merge_tagged(out: list[Candidate], pages: list[dict[str, Any]]) -> None:
    """The headings a tagged PDF declares (its H1-H6): the line they name gets the tag, and one with no line of its
    own (a title LiteParse merged into a block) is added."""
    heads = [h["text"] for page in pages for h in page.get("tagged_headings") or []]
    if not heads:
        return
    shaped = sum(1 for t in heads if len(t) <= 110 and (t[:1].isupper() or t[:1].isdigit())
                 and not t.rstrip().endswith((".", ",", ";")) and not t.lstrip().startswith(("•", "-")))
    if shaped < 0.85 * len(heads):
        return  # the tags of this PDF mark body text as headings too: they tell nothing
    for page in pages:
        page_no = int(page.get("page_index") or 0)
        on_page = [c for c in out if c.page == page_no]
        for k, head in enumerate(page.get("tagged_headings") or []):
            text, level = head["text"], int(head.get("level") or 1)
            match = max(((similar(c.text, text), c) for c in on_page), key=lambda x: x[0], default=(0.0, None))
            if match[1] is not None and match[0] >= 0.8:
                c = match[1]
                c.sources.add("tagged")
                c.heading_block = True
                c.md = c.md or level
                continue
            _candidate(out, page_no, _order_in_page(pages, page_no, text) if page_no <= len(pages) else k, text,
                       md=level, heading_block=True, source="tagged", bare=True)


def _noise(c: Candidate, repeated: set[str]) -> Optional[str]:
    if is_repeated(c, repeated):
        return "repeated"
    if c.leaders:
        return "index_entry"
    if tableish(c.text) and c.mark is None:
        return "table"
    if LETTER_ITEM.match(c.text):
        return "letter_item"
    if len(c.text) < 3 or (c.mark is None and len(c.text) > MAX_TEXT):
        return "length"
    if re.match(r"^\d{1,3}(?:\.\d{3})+(?:,\d+)?\b", c.text):
        return "amount"
    return None


def index_pages(cands: list[Candidate], repeated: set[str], total_pages: int) -> set[int]:
    """Pages of the document's own table of contents: five lines with leaders, or a run of four or more titles that
    come back later in the body in the same order (needing neither leaders nor page numbers)."""
    by_page: dict[int, list[Candidate]] = {}
    for c in cands:
        by_page.setdefault(c.page, []).append(c)
    found = {p for p, cs in by_page.items() if sum(c.leaders for c in cs) >= 5}
    horizon = max(3, int(total_pages * 0.25))
    for p, cs in by_page.items():
        if p > horizon or p in found:
            continue
        titled = [c for c in cs if (c.mark or c.heading_block) and not is_repeated(c, repeated)]
        if len(titled) < 4:
            continue
        later = [c for c in cands if c.page > p]
        cursor, hits = 0, 0
        for c in titled:
            key = (strip_num(c.text) or norm(c.text))[:40]
            if len(key) < 6:
                continue
            for j in range(cursor, len(later)):
                other = (strip_num(later[j].text) or norm(later[j].text))[:40]
                if other.startswith(key[:30]) or key.startswith(other[:30]) and len(other) >= 12:
                    hits, cursor = hits + 1, j + 1
                    break
        if hits >= 4 and hits >= 0.6 * len(titled):
            found.add(p)
    return found


def strong(c: Candidate, body: Optional[float]) -> bool:
    """Looks like a heading whatever its numbering says: the reader flagged it, the layout or the index named it,
    it is bold, in capitals, or set larger than the body."""
    title = c.mark.rest if c.mark and c.mark.rest else c.text
    letters = [ch for ch in title if ch.isalpha()]
    caps = bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.7 and len(title.split()) <= 16
    larger = bool(c.size and body and c.size >= body + 0.9)
    short = len(c.text) <= 120 and not c.text.rstrip().endswith((":", ";", ","))
    return bool(c.heading_block or "layout" in c.sources or "toc" in c.sources or "tagged" in c.sources or larger
                or ((c.bold or caps) and short))


HISTORY = 40            # accepted positions per family a line may continue


def _history(state: dict, family: str) -> list:
    return state.setdefault(family, [])


def _continues(state: dict[str, list], m: Marker) -> bool:
    """Continues the most recent position of its family, or one of the earlier ones: a list taken for sections
    («1.», «2.» inside a clause) must not hide that «9.» follows «8.4»."""
    history = _history(state, m.family)[-HISTORY:]
    if not history:
        return False
    latest = history[-1]
    # an earlier position only counts when the numbering moves forward: «9.» after a stray list «1., 2., 3.» that
    # followed «8.4», never «2.» after the sequence has reached «3».
    return any(follows(prev, m.path) and (prev == latest or m.path > latest) for prev in reversed(history))


def _gap(state: dict[str, list], m: Marker) -> bool:
    for last in reversed(_history(state, m.family)[-HISTORY:]):
        if not m.path:
            return False
        d = len(m.path)
        if d > len(last) + 1:
            continue
        if d == len(last) + 1 and m.path[:-1] == last and 1 < m.path[-1] <= 1 + GAP_MAX:
            return True
        if d >= 2 and m.path[-1] == 1 and follows(last, m.path[:-1]):
            return True  # «4.1» after «3.2»: the «4.» heading was lost by the reader
        if d <= len(last) and m.path[:-1] == last[: d - 1] and last[d - 1] < m.path[-1] <= last[d - 1] + 1 + GAP_MAX:
            return True
    return False


def _confirmed(cands: list[Candidate], i: int, m: Marker, body: Optional[float], need_strong: bool) -> bool:
    for c in cands[i + 1:i + 1 + LOOKAHEAD]:
        other = c.mark
        if other and m.family == "letter" and other.family == "roman" and len(other.raw) == 1:
            other = Marker("letter", (ord(other.raw) - ord("A") + 1,), other.raw, other.rest)  # «H.» then «I.»
        if other and other.family == m.family and follows(m.path, other.path):
            return strong(c, body) or not need_strong
    return False


def _as_letter(state: dict[str, list], m: Marker) -> Optional[Marker]:
    """«I.» after «H.», «V.» after «U.», «C.» after «B.»: a letter of a lettered sequence, not a roman chapter."""
    if m.family != "roman" or len(m.raw) != 1:
        return None
    letters = [p for p in _history(state, "letter") if len(p) >= 1]
    value = ord(m.raw) - ord("A") + 1
    if letters and letters[-1][0] + 1 == value:
        return Marker("letter", (value,), m.raw, m.rest)
    return None


def decide(cands: list[Candidate], body: Optional[float]) -> None:
    """The numbering spine: sets ``decision`` ("heading" / "body" / None for the judge), ``reason`` and ``seq`` of
    every numbered candidate; unnumbered ones are left to the judge and the rules."""
    state: dict[str, list] = {}
    for i, c in enumerate(cands):
        if c.decision is not None or c.mark is None:
            continue
        m = _as_letter(state, c.mark) or c.mark
        c.mark = m
        looks = strong(c, body)
        flagged = c.heading_block or "layout" in c.sources or "toc" in c.sources or "tagged" in c.sources
        if m.family in ("annex", "chapter") and not m.path:  # «ANEXOS», «CAPÍTULO» without a number
            c.seq = "start"
            c.decision, c.reason = ("heading", "spine") if looks else (None, "")
            continue
        history = _history(state, m.family)
        if _continues(state, m):
            c.seq = "continue"
        elif not history and (first_of_sequence(m.path) or m.family in CLAUSE_FAMILIES + TOP_FAMILIES
                              or (c.sources == {"table"} and len(m.path) == 1)):
            c.seq = "start"
        elif first_of_sequence(m.path) or m.family == "annex":
            c.seq = "restart"
        elif _gap(state, m):
            c.seq = "gap"
        else:
            c.seq = "break"
        accept: Optional[bool]
        scope = state.get("_clause")
        if scope and m.family in ("decimal", "paren") and len(m.path) == 1:
            # «Clàusula 2» then «1. El contracte es tipifica…», «2. Atenent…»: the clause's own numbered paragraphs.
            c.seq = "clause_item"
            if not _title_shaped(c):
                c.decision, c.reason = "body", "spine:clause_list"
            # a short numbered line inside a clause may be its sub-section or an item of its list: the judge reads it
            continue
        if c.seq == "continue":
            accept = True if not (c.inline and len(m.rest) < 3) else None
        elif c.seq == "start":
            after_reset = state.get("_reset") and m.family in ("decimal", "paren")
            if m.family in ("decimal", "paren") and len(m.path) == 1 and _sentence(c):
                accept = False  # a sequence never starts with a sentence: that is a list
            elif flagged or (_confirmed(cands, i, m, body, need_strong=False) and (not after_reset or _title_shaped(c))):
                accept = True
            else:
                accept = None if looks else False
        elif c.seq == "restart":
            if m.family == "annex":
                accept = True if looks else None
            elif m.family in ("decimal", "paren") and len(m.path) == 1 and _sentence(c):
                accept = False
            elif flagged and _confirmed(cands, i, m, body, need_strong=True):
                accept = True
            else:
                accept = None if (flagged or looks) and not c.inline else False
        elif c.seq == "gap":
            accept = True if (looks or not c.inline) else None
        else:
            if "toc" in c.sources and ("layout" in c.sources or c.heading_block):
                accept = True  # the document's index and its layout both name it
            else:
                accept = None if looks and not c.inline else False
        if accept is True:
            c.decision, c.reason = "heading", f"spine:{c.seq}"
            _advance(state, m, c.seq)
        elif accept is False:
            c.decision, c.reason = "body", f"spine:{c.seq}"


def _since(state: dict[str, list], scope: tuple, m: Marker) -> bool:
    """The depth-1 decimal sequence restarted after the clause opened (its positions are not the clause's)."""
    clause_value = scope[0]
    return m.path[0] != clause_value or len(m.path) == 1 and state.get("_clause_restarted")


def _sentence(c: Candidate) -> bool:
    """«1. Que consto/l'empresa consta inscrita…», «2. La proposició presentada per les empreses…»: the text of the
    line reads as a sentence (a declaration, a requirement), whatever its format says."""
    rest = (c.mark.rest if c.mark else c.text).strip(" *_")
    if re.match(r"(?i)^(que|qu[eè]|si|en el cas|en caso|cuando|quan|perqu[eè]|porque)\b", rest):
        return True
    head = re.split(r"(?<=[a-zà-ÿ]{3})[.:](?=\s|$)", rest, maxsplit=1)[0]
    return len(head) > 90 or bool(re.search(
        r"\b(?:és|es|son|són|serà|será|seran|serán|haurà|deberá|podrà|podrá|presentada|presentades|consta|disposa|"
        r"declara|declaro|assumeix|asume|té|tiene)\b", head, re.I))


def _title_shaped(c: Candidate) -> bool:
    """«1. Objecte del contracte», «2. **Règim jurídic** El contracte…»: a short title or a bold one ahead of the
    text; not «1. Aquest contracte té per objecte el subministrament…»."""
    rest = c.mark.rest if c.mark else c.text
    if c.bold or c.heading_block or "layout" in c.sources or "toc" in c.sources:
        return True
    head = re.split(r"(?<=[a-zà-ÿ]{3})[.:](?=\s|$)", rest, maxsplit=1)[0]
    letters = [ch for ch in head if ch.isalpha()]
    caps = bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.7
    return caps or (len(head) <= 70 and len(head.split()) <= 9 and not re.search(
        r"\b(?:és|es|son|són|serà|será|haurà|deberá|podrà|podrá|se|s'|que)\b", head, re.I))


def _advance(state: dict[str, list], m: Marker, seq: str) -> None:
    if m.family in CLAUSE_FAMILIES and m.path:
        state["_clause"] = m.path
        state["_clause_restarted"] = False
    elif m.family == "decimal" and len(m.path) == 1 and state.get("_clause") and m.path[0] == 1:
        state["_clause_restarted"] = True
    if m.family in ("annex", "chapter"):
        state["_reset"] = m.family == "annex"
        # a new annex or chapter: the numbering below begins again (a roman list inside a section does not)
        for family in list(state):
            if family.startswith("_") and family != "_reset":
                state[family] = None
            elif family not in TOP_FAMILIES:
                state[family] = []
    _history(state, m.family).append(m.path)
    if m.family in CLAUSE_FAMILIES and m.path:
        _history(state, "decimal").append((m.path[0],))  # «Novena» (or «Cláusula 9») is the parent of «9.1»


def ranks(headings: list[Candidate]) -> None:
    """Each heading's rank in the document's hierarchy: grouping keywords in their own order («Título» over
    «Capítulo» over «Sección»), roman chapters and annexes at the top, then clauses («Primera», «Cláusula 3», «3»),
    their decimal sub-sections, and an unnumbered title one rank under the section it sits in. Ranks are compacted
    to the ones the document uses."""
    from .numbering import keyword_rank
    families = {c.mark.family for c in headings if c.mark}
    first_clause = next((k for k, c in enumerate(headings) if c.mark and c.mark.family in CLAUSE_FAMILIES + ("decimal",)),
                        len(headings))
    roman_top = any(c.mark and c.mark.family == "roman" for c in headings[:first_clause + 1])
    if not roman_top:
        families.discard("roman")
    has_top = bool(families & {"chapter", "roman"})
    group_ranks = sorted({keyword_rank(c.mark.raw) for c in headings if c.mark and c.mark.family == "chapter"})
    depth_of_group = {r: k for k, r in enumerate(group_ranks)}
    clause_rank = len(group_ranks) if group_ranks else (1 if "roman" in families else 1)
    if "roman" in families and group_ranks:
        clause_rank = len(group_ranks)
    for k, c in enumerate(headings):
        m = c.mark
        if m and m.family == "chapter":
            c.rank = depth_of_group[keyword_rank(m.raw)]
        elif m and m.family == "annex":
            c.rank = 0
        elif m and m.family == "roman":
            c.rank = 0 if roman_top else None
        elif m and m.family in CLAUSE_FAMILIES:
            c.rank = clause_rank
        elif m and m.family in ("decimal", "letter", "paren"):
            c.rank = clause_rank - 1 + len(m.path) if has_top else len(m.path)
        else:
            nxt = next((h for h in headings[k + 1:k + 3] if h.mark), None)
            letters = [ch for ch in c.text if ch.isalpha()]
            caps = bool(letters) and sum(ch.isupper() for ch in letters) / len(letters) > 0.7
            opens = nxt is not None and nxt.mark.family not in TOP_FAMILIES and nxt.page - c.page <= 1 and (
                nxt.seq in ("start", "restart")
                or ((caps or (c.heading_block and c.md <= 2)) and nxt.mark.family in CLAUSE_FAMILIES + ("decimal", "letter")
                    and len(nxt.mark.path) == 1 and any(h.rank == 0 and not h.mark for h in headings[:k])))
            c.rank = 0 if (opens and not has_top) else None
        if c.rank is not None and not has_top and m and m.family not in ("annex",):
            c.rank = max(1, c.rank)


def build(headings: list[Candidate], total_pages: int) -> list[dict[str, Any]]:
    ordered = sorted(headings, key=lambda c: (c.page, c.order))
    nodes: list[dict[str, Any]] = []
    stack: list[tuple[int, dict]] = []  # (rank, node)
    roots: list[dict[str, Any]] = []
    section = 0  # the rank of the last heading the rules ranked: the section the unranked ones sit in
    for c in ordered:
        rank = c.rank
        if rank is None:
            # One under that section, as siblings of each other: under the previous heading instead, a run of
            # unnumbered titles (a form, an announcement) nests one level deeper at each.
            rank = section + 1
        else:
            section = rank
        while stack and stack[-1][0] >= rank:
            stack.pop()
        node = {"title": _title(c), "start_index": c.page, "end_index": c.page, "nodes": [], "_rank": rank}
        (stack[-1][1]["nodes"] if stack else roots).append(node)
        stack.append((rank, node))
        nodes.append(node)

    def spans(items: list[dict], end: int) -> None:
        for k, n in enumerate(items):
            following = items[k + 1]["start_index"] if k + 1 < len(items) else end
            n["end_index"] = max(n["start_index"], following)
            spans(n["nodes"], n["end_index"])

    spans(roots, total_pages)

    def clean(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"title": n["title"], "start_index": n["start_index"], "end_index": n["end_index"],
                 **({"nodes": clean(n["nodes"])} if n["nodes"] else {})} for n in items]
    return clean(roots)


def _title(c: Candidate) -> str:
    """The heading as it reads, cut where a numbered paragraph's body starts: «9.2 La capacitat d'obrar de les
    empreses…» keeps its first clause, not the paragraph."""
    text = c.text
    if len(text) <= 120:
        return text
    cut = re.search(r"[.:;](?=\s)", text[20:120])
    return text[: 20 + cut.start() + 1] if cut else text[:120].rsplit(" ", 1)[0] + "…"


def collect(pages: list[dict[str, Any]], layout_structure: Optional[list[dict]] = None) -> tuple[list[Candidate], dict[str, Any], Optional[float]]:
    lines = lines_of(pages)
    repeated = repeated_texts(lines)
    toc_pages = index_pages(lines, repeated, len(pages))
    # an index repeats the titles it lists: what repeats is decided without its pages (and one line per page)
    seen_lines: set = set()
    body_lines = []
    for c in lines:
        key = (c.page, norm(c.text))
        if c.page not in toc_pages and key not in seen_lines:
            seen_lines.add(key)
            body_lines.append(c)
    repeated = repeated_texts(body_lines)
    body = _body_size(pages)
    kept: list[Candidate] = []
    dropped: Counter = Counter()
    for c in lines:
        if c.page in toc_pages:
            dropped["index_page"] += 1
            continue
        why = _noise(c, repeated)
        if why:
            dropped[why] += 1
            continue
        kept.append(c)
    entries = [c for c in lines if c.page in toc_pages and (c.mark or c.heading_block)]
    for page_no, order, title in _located_index_entries(entries, pages, max(toc_pages, default=0)):
        match = _closest(kept, page_no, title)
        if match:
            match.sources.add("toc")
    for node in _flatten(layout_structure or []):
        title = node["title"]
        if title.lower().startswith(("page ", "preface")) or title.endswith("(intro)") or len(title) > MAX_TEXT:
            continue
        match = _closest(kept, node["page"], title)
        if match:
            match.sources.add("layout")
    kept.sort(key=lambda c: (c.page, c.order))
    for i, c in enumerate(kept):
        c.index = i
    _drop_index_copies(kept, len(pages))
    decide(kept, body)
    recovered = recover_gaps(kept, pages)
    if recovered:
        kept.sort(key=lambda c: (c.page, c.order))
        for i, c in enumerate(kept):
            c.index = i
    # The cover: unnumbered lines of the first page before the first numbered section are the document's title.
    first = next((c for c in kept if c.decision == "heading" and c.mark is not None), None)
    if first is not None and first.page > 1:
        for c in kept:
            if c.page == 1 and c.decision is None and c.mark is None and "toc" not in c.sources:
                c.decision, c.reason = "body", "cover"
    # Containers: an unnumbered heading the reader flagged that opens a sequence right below it («CUADRO DE
    # CARACTERÍSTICAS DEL CONTRATO» before its «1.»).
    for k, c in enumerate(kept):
        if c.decision is not None or c.mark is not None or not (c.heading_block or c.bold):
            continue
        nxt = next((h for h in kept[k + 1:k + 4] if h.mark is not None), None)
        if nxt is not None and nxt.decision == "heading" and nxt.seq in ("start", "restart") and nxt.page - c.page <= 1 \
                and len(c.text) <= 90 and not c.text.rstrip().endswith((":", ",", ".")):
            c.decision, c.reason = "heading", "container"
    seen: dict[str, int] = {}
    for c in kept:
        if c.decision == "heading" and c.marker:
            c.repeats = c.marker in seen
            seen.setdefault(c.marker, c.index)
        c.known = c.decision is not None
    report = {"lines": len(lines), "candidates": len(kept), "known": sum(c.known for c in kept),
              "index_pages": sorted(toc_pages), "dropped": dict(dropped), "body_size": body,
              "repeated": sorted(k for k in repeated if " " in k)[:20]}
    return kept, report, body


def _drop_index_copies(cands: list[Candidate], total_pages: int) -> None:
    """A numbered clause, chapter or annex that comes back later in the body was, in the first pages, an entry of the
    document's index (one the leaders did not give away): content, never the section."""
    horizon = max(3, int(total_pages * 0.12))
    later: dict[tuple, int] = {}
    for c in cands:
        if c.mark and c.mark.family in ("clause", "ordinal") and c.mark.path:
            later[(c.mark.family, c.mark.path)] = max(later.get((c.mark.family, c.mark.path), 0), c.page)
    for c in cands:
        if c.mark and c.page <= horizon and c.decision is None and c.mark.path \
                and later.get((c.mark.family, c.mark.path), 0) > c.page + 1:
            c.decision, c.reason, c.seq = "body", "index_copy", "index"


def _marker_pattern(family: str, path: tuple, bare: bool) -> Optional[str]:
    """How the missing number would be written at the start of a line: «8.4», «8.4.», «B.2», «B2», «3 OBJETO»."""
    if family == "decimal":
        num = r"\.".join(str(x) for x in path)
        sep = r"(?:\.?\s*[-–)]?\s*|\s+)" if len(path) > 1 or not bare else r"(?:\s*[.\-–)]\s*|\s+(?=[A-ZÀ-ÚÑÇ]{2}))"
        return rf"(?<![\d.,]){num}{sep}(?=[A-ZÀ-ÚÑÇa-zà-ÿ¿«\"'])"
    if family == "letter":
        head = chr(ord("A") + path[0] - 1)
        tail = "".join(rf"\.?{x}" for x in path[1:])
        return rf"(?<![A-Za-z\d]){head}{tail}\s*[.\-–)]\s*(?=[A-ZÀ-ÚÑÇ])"
    return None


def recover_gaps(cands: list[Candidate], pages: list[dict[str, Any]]) -> list[Candidate]:
    """The numbers a sequence skips: between «8.3» and «8.5» there is an «8.4», between rows «2» and «4» a «3».
    Each one is looked for at the start of a line or after a full stop in the text between its two neighbours, and
    added as a heading when it is found once."""
    accepted = [c for c in cands if c.decision == "heading" and c.mark and c.mark.family in ("decimal", "letter")]
    added: list[Candidate] = []
    texts = {int(p.get("page_index") or 0): p.get("markdown") or "" for p in pages}
    by_family: dict[str, list[Candidate]] = {}
    for c in accepted:
        by_family.setdefault(c.mark.family, []).append(c)
    for family, seq in by_family.items():
        first = seq[0]
        if len(first.mark.path) == 1 and 2 <= first.mark.path[0] <= 4:
            # the sequence's first rows, lost before it started: look back up to two pages
            phantom = Candidate(index=-1, page=max(1, first.page - 2), order=-1, text="", mark=Marker(
                family, (0,), "", ""), sources=set())
            seq = [phantom] + seq
        for a, b in zip(seq, seq[1:]):
            pa, pb = a.mark.path, b.mark.path
            missing: list[tuple] = []
            if pa == (0,):
                missing = [(v,) for v in range(1, pb[0])]
            elif len(pa) == len(pb) and pa[:-1] == pb[:-1] and 1 < pb[-1] - pa[-1] <= 1 + GAP_MAX:
                missing = [(*pa[:-1], v) for v in range(pa[-1] + 1, pb[-1])]
            elif len(pb) == len(pa) + 1 and pb[:-1] == pa and 1 < pb[-1] <= 1 + GAP_MAX:
                missing = [(*pa, v) for v in range(1, pb[-1])]
            if not missing or b.page - a.page > 3:
                continue
            bare = a.sources == {"table"} or b.sources == {"table"} or (len(pb) == 1 and not b.mark.raw.endswith("."))
            for path in missing:
                pattern = _marker_pattern(family, path, bare)
                if not pattern:
                    continue
                hits = []
                for page_no in range(a.page, b.page + 1):
                    text = texts.get(page_no, "")
                    start = text.find(a.text[:30]) + len(a.text[:30]) if a.text and page_no == a.page and a.text[:30] in text else 0
                    end = text.find(b.text[:30]) if page_no == b.page and b.text[:30] in text else len(text)
                    for m in re.finditer(r"(?:^|\n|(?<=[.:;|])\s+|\|\s*)(?:[#*_\s]*)(" + pattern + ")", text[start:end]):
                        line = text[start + m.start(1):start + m.start(1) + 160].split("\n")[0]
                        hits.append((page_no, re.sub(r"[*_|#]+", " ", line).strip()))
                if len(hits) != 1:
                    continue
                page_no, line = hits[0]
                line = re.sub(r"\s+", " ", line)
                mark = parse(line, bare=True)
                if not mark or mark.path != path:
                    continue
                order = (a.order + 1) if page_no == a.page else (b.order - 1 if page_no == b.page else 0)
                c = Candidate(index=-1, page=page_no, order=order, text=line[:400], marker=mark.raw, numbered=True,
                              sources={"gap"}, mark=mark, seq="recovered", decision="heading", reason="spine:recovered",
                              known=True)
                cands.append(c)
                added.append(c)
    return added


def decide_unsettled(c: Candidate, body: Optional[float], numbered: bool = False) -> str:
    """Without a judge, the lines the spine left open: a heading when it looks like one, else content. In a
    numbered document an unnumbered line needs the index, or the layout and the reader's flag together."""
    if c.mark is not None:
        if c.seq == "clause_item":
            return "heading" if (c.heading_block or "layout" in c.sources or "toc" in c.sources) else "body"
        return "heading" if strong(c, body) and c.seq in ("start", "gap", "restart") else "body"
    if ("toc" in c.sources or "tagged" in c.sources) and len(c.text) <= MAX_TEXT:
        return "heading"  # the document's index or its own heading tags name it
    shaped = len(c.text) <= 100 and not c.text.rstrip().endswith((":", ",", ".", ";"))
    if numbered:
        votes = sum(("layout" in c.sources, "inspector" in c.sources, c.heading_block and c.md <= 3, c.bold))
        return "heading" if shaped and votes >= 2 and ("layout" in c.sources or "inspector" in c.sources) else "body"
    if "layout" in c.sources and len(c.text) <= MAX_TEXT:
        return "heading"
    if c.heading_block and c.md <= 3 and 8 <= len(c.text) <= MAX_TEXT and len(c.text.split()) >= 2 \
            and not c.text.rstrip().endswith((":", ",")):
        return "heading"
    if c.bold and strong(c, body) and 8 <= len(c.text) <= 100 and len(c.text.split()) >= 2:
        return "heading"
    return "body"
