"""Mode ``headings``: the tree of a PDF from its own headings. LiteParse's heading blocks and the document's
numbering give the candidates and settle most of them; the document's index pages, located in the body, and the
layout detection of flash add what the reader missed; a judge (Jev) decides the lines no rule settles; the rest
of the pipeline (summaries, description, storage) is the flash one. No optimizer: a short section stays a
section, because that is how the document is asked for."""
from __future__ import annotations

import logging
from typing import Any, Optional

from .builder import build_structure
from .candidates import Candidate, collect, decide_without_judge
from .judge import TYPESAFE_MODEL, TYPESAFE_URL, HeadingJudge, TypesafeHeadingJudge

logger = logging.getLogger(__name__)

__all__ = ["build_headings_structure", "Candidate", "HeadingJudge", "TypesafeHeadingJudge", "collect",
           "TYPESAFE_MODEL", "TYPESAFE_URL"]


# The judge adds a heading the rules missed with a fair margin, and takes one away only when sure: the rules
# alone recover 98 % of the headings of real specifications, so the judge is there for the remaining lines.
JUDGE_ADDS_AT = 0.6
JUDGE_REMOVES_AT = 0.85


def _settle(rule: str, choice: str, probability: float) -> tuple[str, str]:
    if choice == rule:
        return rule, f"judge:{probability:.2f}"
    if choice == "heading" and probability >= JUDGE_ADDS_AT:
        return "heading", f"judge:{probability:.2f}"
    if choice == "body" and probability >= JUDGE_REMOVES_AT:
        return "body", f"judge:{probability:.2f}"
    return rule, f"rule (judge {choice} {probability:.2f})"


def build_headings_structure(pages: list[dict[str, Any]], page_texts: list[str],
                             layout_structure: Optional[list[dict]] = None,
                             judge: Optional[HeadingJudge] = None,
                             document: Optional[dict[str, Any]] = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The structure and a report of how it was built: candidates, how many the numbering spine settled, how many
    the judge decided, how many fell to the rules, and the headings kept."""
    from ..flash.api import _add_intros, _add_preface, _page_lines
    from .structure import build, collect, decide_unsettled, ranks
    candidates, report, body = collect(pages, layout_structure)
    ask = [c for c in candidates if c.decision is None]
    judged: dict[int, tuple[str, float]] = {}
    if judge is not None and ask:
        try:
            judged = judge.decide(document or {}, candidates, ask)
        except Exception as exc:  # the rules decide; indexing goes on
            logger.warning("headings judge unavailable, deciding by rules: %s", exc)
            report["judge_error"] = str(exc)[:200]
    numbered = sum(1 for c in candidates if c.decision == "heading" and c.mark is not None) >= 8
    for c in ask:
        rule = decide_unsettled(c, body, numbered)
        if c.index not in judged:
            c.decision, c.reason = rule, "rule"
            continue
        choice, probability = judged[c.index]
        c.decision, c.reason = _settle(rule, choice, probability)
    headings = [c for c in candidates if c.decision == "heading"]
    if judge is not None and getattr(judge, "verify_pass", False) and headings:
        accepted = {(c.mark.family, c.mark.path) for c in headings if c.mark is not None}

        def in_run(c) -> bool:
            """A first-level number with both neighbours accepted («3.-» between «2.-» and «4.-»): a whole sequence."""
            m = c.mark
            if m is None or len(m.path) != 1:
                return False
            before, after = (m.family, (m.path[0] - 1,)), (m.family, (m.path[0] + 1,))
            return before in accepted and after in accepted

        for c in headings:
            m = c.mark
            c._verify = not ("toc" in c.sources or "tagged" in c.sources or c.reason == "container"
                             or (m is not None and m.family in ("chapter", "annex", "clause", "ordinal"))
                             or (m is not None and len(m.path) >= 2 and c.seq in ("continue", "recovered"))
                             or in_run(c))
        try:
            verdicts = judge.verify(document or {}, headings)
        except Exception as exc:
            logger.warning("headings verify unavailable: %s", exc)
            verdicts = {}
        removed = [c for c in headings if verdicts.get(c.index, ("", 0.0))[0] == "body"
                   and verdicts[c.index][1] >= JUDGE_REMOVES_AT]
        for c in removed:
            c.decision, c.reason = "body", f"verify:{verdicts[c.index][1]:.2f}"
        headings = [c for c in headings if c.decision == "heading"]
        report["verified"] = sum(1 for c in candidates if getattr(c, "_verify", False))
        report["verify_removed"] = len(removed)
    ranks(headings)
    structure = build(headings, len(page_texts))
    if structure:
        lines = _page_lines(page_texts)
        if structure[0]["start_index"] > 1:
            _add_preface(structure, lines)
        _add_intros(structure, lines)
    report.update({
        "asked": len(ask), "judged": len(judged),
        "by_rule": sum(1 for c in ask if c.reason == "rule"),
        "headings": len(headings),
        "judge": getattr(judge, "name", None) if judge is not None else None,
        "judge_usage": getattr(judge, "usage", None) if judge is not None else None,
        "decisions": [{"page": c.page, "text": c.text[:80], "decision": c.decision, "reason": c.reason,
                       "seq": c.seq, "rank": c.rank, "sources": sorted(c.sources)} for c in candidates],
    })
    return structure, report
