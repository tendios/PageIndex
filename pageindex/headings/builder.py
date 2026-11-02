"""From decided headings to a PageIndex structure: levels from the document's own numbering, page ranges that
run to the next heading of the same or a higher level, and the preface and intro nodes flash mode adds."""
from __future__ import annotations

from typing import Any

from .candidates import ANNEX_WORDS, CHAPTER_WORDS, Candidate


def level_of(c: Candidate, has_chapters: bool) -> int:
    """Chapters, titles and annexes open the first level; clauses and numbered sections hang from them when the
    document has chapters, and open the first level otherwise; `1.2` is one deeper than `1`."""
    if c.keyword in CHAPTER_WORDS or c.keyword in ANNEX_WORDS:
        return 1
    shift = 1 if has_chapters else 0
    if c.keyword:
        return 1 + shift
    if c.marker and c.marker[:1].isdigit():
        return min(4, c.marker.count(".") + 1 + shift)
    return min(4, max(1, c.md or 1) + shift)


def build_structure(headings: list[Candidate], total_pages: int) -> list[dict[str, Any]]:
    """The nested tree of the headings, in document order, each spanning until the next heading of its level or
    above (pages may be shared, as the index's own trees do)."""
    ordered = sorted(headings, key=lambda c: (c.page, c.order))
    has_chapters = any(c.keyword in CHAPTER_WORDS for c in ordered)
    nodes = [{"title": c.text, "start_index": c.page, "end_index": c.page, "level": level_of(c, has_chapters), "nodes": []}
             for c in ordered]
    for i, node in enumerate(nodes):
        following = next((m for m in nodes[i + 1:] if m["level"] <= node["level"]), None)
        node["end_index"] = following["start_index"] if following else total_pages
    roots: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for node in nodes:
        while stack and stack[-1]["level"] >= node["level"]:
            stack.pop()
        (stack[-1]["nodes"] if stack else roots).append(node)
        stack.append(node)

    def clean(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [{"title": n["title"], "start_index": n["start_index"], "end_index": n["end_index"],
                 **({"nodes": clean(n["nodes"])} if n["nodes"] else {})} for n in items]
    return clean(roots)
