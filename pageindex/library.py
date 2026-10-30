"""Library view: several documents' outlines in one payload, so a
multi-document question starts from every outline at once instead of one
get_document_structure() call per document. Local mode only.

A view is a list of document roots — doc_name, doc_id, page_count,
description, metadata — each carrying its outline under ``nodes`` (titles,
page ranges and node summaries, as get_document_structure() returns them,
without text). An optional topic index (``build_topic_index``) maps topics
to sections across the library.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Callable, Optional

LIBRARY_VIEW_MAX_TOKENS = 12_000
TOPIC_INDEX_VERSION = 1
_TOPIC_BATCH_CHARS = 60_000
_TOPIC_SUMMARY_CHARS = 300


def document_root(client, entry: dict) -> dict:
    """One document's root: its listing fields and formatted outline."""
    from .agent_tools import _format_structure
    tree = client.get_tree(entry["id"], node_summary=True,
                           include_text=False).get("result") or []
    root: dict[str, Any] = {
        "doc_name": entry.get("name"),
        "doc_id": entry.get("id"),
        "page_count": entry.get("pageNum"),
        "description": entry.get("description"),
    }
    if entry.get("metadata"):
        root["metadata"] = entry["metadata"]
    root["nodes"] = _format_structure(tree)
    return root


def _count_nodes(nodes) -> int:
    return sum(1 + _count_nodes(node.get("nodes") or [])
               for node in nodes or [] if isinstance(node, dict))


def prune(nodes: list, levels: int) -> list:
    """The outline down to ``levels`` deep; a cut node says how many
    descendants it hides (``nodes_omitted``)."""
    pruned = []
    for node in nodes or []:
        if not isinstance(node, dict):
            continue
        node = dict(node)
        children = node.get("nodes") or []
        if children:
            if levels <= 1:
                del node["nodes"]
                node["nodes_omitted"] = _count_nodes(children)
            else:
                node["nodes"] = prune(children, levels - 1)
        pruned.append(node)
    return pruned


def _abridged(roots: list, levels: int) -> list:
    out = []
    for root in roots:
        root = dict(root)
        if levels:
            root["nodes"] = prune(root.get("nodes") or [], levels)
        else:
            root["nodes_omitted"] = _count_nodes(root.pop("nodes", None))
        out.append(root)
    return out


def fit_view(roots: list, max_tokens: int,
             count: Callable[[str], int]) -> "tuple[list, Optional[int]]":
    """The roots within ``max_tokens``: complete when they fit, else levels
    1-2, then level 1, then names and metadata only. Returns (view, levels
    kept — None when complete)."""
    for levels in (None, 2, 1, 0):
        view = roots if levels is None else _abridged(roots, levels)
        if count(json.dumps(view, ensure_ascii=False)) <= max_tokens:
            return view, levels
    return view, 0


# ── topic index ──

def _iter_nodes(nodes, depth=1):
    for node in nodes or []:
        if isinstance(node, dict):
            yield node, depth
            yield from _iter_nodes(node.get("nodes"), depth + 1)


def _parse_topics(text: str) -> dict:
    match = re.search(r"\{.*\}", text or "", re.S)
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except ValueError:
        return {}
    topics = data.get("topics") if isinstance(data, dict) else None
    return topics if isinstance(topics, dict) else {}


_TOPIC_PROMPT = """\
You are building a topic index over a library of documents, so a reader can \
jump from a topic to every section that covers it, across documents.

Sections, one per line: key | document | section title | pages | summary
{sections}

Group the sections under short, specific topics (2-6 words) a reader would \
look up, e.g. "revenue by segment", "manufacturing partners". A section may \
sit under several topics; skip boilerplate (cover pages, tables of contents, \
signatures). Prefer topics that span more than one document.{existing}

Return JSON only: {{"topics": {{"<topic>": ["<key>", ...]}}}}"""


def build_topic_index(roots: list, complete: Callable[[str], str],
                      model: Optional[str] = None,
                      created_at: Optional[dict] = None) -> dict:
    """Topic → sections across ``roots`` (document roots with outlines),
    built by ``complete(prompt) -> text`` in batches of node summaries;
    later batches are shown the topics so far, to reuse their names."""
    sections = []
    for root in roots:
        for node, _ in _iter_nodes(root.get("nodes")):
            if node.get("node_id") is None:
                continue
            summary = " ".join(str(node.get("summary")
                                   or node.get("prefix_summary") or "").split())
            sections.append((root, node, summary[:_TOPIC_SUMMARY_CHARS]))
    batches, batch, size = [], [], 0
    for key, (root, node, summary) in enumerate(sections):
        line = (f"{key} | {root.get('doc_name')} | {node.get('title', '')} | "
                f"{node.get('start_index')}-{node.get('end_index')} | {summary}")
        if batch and size + len(line) > _TOPIC_BATCH_CHARS:
            batches.append(batch)
            batch, size = [], 0
        batch.append(line)
        size += len(line) + 1
    if batch:
        batches.append(batch)

    topics: dict[str, list[dict]] = {}
    for lines in batches:
        existing = (
            "\nReuse these existing topic names where they fit: "
            + json.dumps(sorted(topics), ensure_ascii=False)
            if topics else "")
        answer = complete(_TOPIC_PROMPT.format(sections="\n".join(lines),
                                               existing=existing))
        for topic, keys in _parse_topics(answer).items():
            topic = " ".join(str(topic).split())
            if not topic or not isinstance(keys, list):
                continue
            entries = topics.setdefault(topic, [])
            for key in keys:
                try:
                    root, node, _ = sections[int(key)]
                except (TypeError, ValueError, IndexError):
                    continue
                entry = {"doc_id": root.get("doc_id"),
                         "node_id": node.get("node_id"),
                         "start_index": node.get("start_index"),
                         "end_index": node.get("end_index")}
                if entry not in entries:
                    entries.append(entry)
    return {
        "version": TOPIC_INDEX_VERSION,
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        # createdAt per document: a replaced or deleted one drops out.
        "documents": created_at or {},
        "topics": {topic: entries for topic, entries in sorted(topics.items())
                   if entries},
    }


def topics_view(index: Optional[dict], entries: list) -> Optional[dict]:
    """The stored index restricted to ``entries`` (listing dicts) and
    still-current documents, keyed by document name for the agent."""
    if not index or index.get("version") != TOPIC_INDEX_VERSION:
        return None
    built = index.get("documents") or {}
    names = {entry.get("id"): entry.get("name") for entry in entries
             if built.get(entry.get("id")) == entry.get("createdAt")}
    view = {}
    for topic, refs in (index.get("topics") or {}).items():
        kept = [{"doc_name": names[ref.get("doc_id")],
                 "node_id": ref.get("node_id"),
                 "pages": f"{ref.get('start_index')}-{ref.get('end_index')}"}
                for ref in refs if ref.get("doc_id") in names]
        if kept:
            view[topic] = kept
    return view or None
