"""The judge of the lines no rule settles: a deterministic classifier (Typesafe's Jev) asked, line by line or
by groups of numbered siblings, whether a line opens a section of this document or is content, with the
document's outline around it. A failed request leaves its lines to the rules; indexing never fails for the
judge."""
from __future__ import annotations

import logging
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional, Protocol

from .candidates import Candidate

logger = logging.getLogger(__name__)

TYPESAFE_URL = "https://api.typesafe.ai/v1"
TYPESAFE_MODEL = "jev-latest"
STRETCH_LINES = 40
IMAGE_STRETCH_LINES = 16
CONTEXT_LINES = 40
MAX_STATE_CHARS = 35_000
OUTLINE_TEXT = 80
CLAUSE_TEXT = 60

CONVENTION = (
    "Un apartado es cualquier parte del documento que alguien puede pedir por su número o su título («redacta la 1.1», "
    "«revisa el anexo II»): un capítulo, una cláusula, una subcláusula o un anexo. Una línea numerada que continúa la "
    "numeración del documento abre un apartado aunque sea un párrafo con texto completo, porque se puede pedir por su "
    "número. También está numerada la línea que empieza por una palabra y un número («Cláusula 22.», «Annex 3»). "
    "state.outline es el tramo del documento donde está la línea: en orden, cada línea que podría ser un título, con su "
    "página (page), su número (marker), el nivel de encabezado que le dio el lector del PDF (md: 1 es el mayor, 0 si no "
    "la marcó como encabezado), si el lector la marcó como encabezado (heading), si la maquetación la detectó como "
    "título (layout), si ya se da por título (known) y si repite el número de una línea anterior (repeats). En "
    "state.lines, previous y following son el texto que hay justo antes y después de la línea en su página. "
    "state.clauses son las líneas con número de primer nivel de todo el documento. Deduce cómo escribe este documento "
    "sus títulos (formato del número, mayúsculas, estilo) y aplica esa convención. Es contenido, no apartado: una lista "
    "que empieza de nuevo dentro de un apartado (a), b), 1., 2. bajo una cláusula), una fila o celda de tabla, un pie o "
    "encabezado de página, una etiqueta o un dato, una frase del cuerpo que empieza por un número, y en un documento "
    "numerado una línea sin número salvo que sea un capítulo, una parte o un anexo que agrupa apartados."
)

CRITERIA = {
    "heading": "Abre un apartado: lo que sigue es su contenido.",
    "body": "Es contenido de un apartado, aunque vaya en negrita, en mayúsculas o numerado.",
}


class HeadingJudge(Protocol):
    name: str

    def decide(self, document: dict[str, Any], candidates: list[Candidate],
               ask: list[Candidate]) -> dict[int, tuple[str, float]]:
        """`{candidate index: (\"heading\" | \"body\", probability)}` for the lines it could decide."""


def outline_line(c: Candidate, text_chars: int = OUTLINE_TEXT) -> dict[str, Any]:
    line: dict[str, Any] = {"index": c.index, "page": c.page, "text": c.text[:text_chars]}
    if c.marker:
        line["marker"] = c.marker
    line["md"] = c.md
    if c.heading_block:
        line["heading"] = True
    if c.bold:
        line["bold"] = True
    if "layout" in c.sources:
        line["layout"] = True
    if c.known:
        line["known"] = True
    if c.repeats:
        line["repeats"] = True
    return line


def line_state(c: Candidate) -> dict[str, Any]:
    state = outline_line(c, text_chars=OUTLINE_TEXT * 2)
    state["previous"] = c.previous
    state["following"] = c.following
    return state


def parent_marker(c: Candidate) -> Optional[str]:
    if c.marker and c.marker[:1].isdigit() and "." in c.marker:
        return c.marker.rsplit(".", 1)[0]
    return None


def sibling_groups(ask: list[Candidate]) -> list[list[Candidate]]:
    """Consecutive numbered siblings (2.1, 2.2, 2.3 under 2) are one question: decided one by one they could split
    a section into a mix of subsections and body text. Everything else is asked alone."""
    groups: list[list[Candidate]] = []
    for c in sorted(ask, key=lambda c: c.index):
        parent = parent_marker(c)
        if groups and parent and parent_marker(groups[-1][-1]) == parent and len(groups[-1]) < 12:
            groups[-1].append(c)
        else:
            groups.append([c])
    return groups


def stretches(groups: list[list[Candidate]], lines_per_request: int = STRETCH_LINES) -> list[list[list[Candidate]]]:
    out: list[list[list[Candidate]]] = []
    current: list[list[Candidate]] = []
    size = 0
    for group in groups:
        if current and size + len(group) > lines_per_request:
            out.append(current)
            current, size = [], 0
        current.append(group)
        size += len(group)
    if current:
        out.append(current)
    return out


def outline_window(candidates: list[Candidate], stretch: list[list[Candidate]],
                   max_chars: int = MAX_STATE_CHARS) -> list[dict[str, Any]]:
    """The outline around a stretch: its lines and their neighbours, fewer when the window would not fit."""
    import json
    indexes = [c.index for group in stretch for c in group]
    first, last = min(indexes), max(indexes)
    window: list[dict[str, Any]] = []
    for context in range(CONTEXT_LINES, -1, -10):
        window = [outline_line(c) for c in candidates[max(0, first - context):last + context + 1]]
        if len(json.dumps(window, ensure_ascii=False)) <= max_chars:
            break
    return window


def clauses(candidates: list[Candidate]) -> list[dict[str, Any]]:
    """The first-level numbered lines of the whole document: its sequence of clauses."""
    return [outline_line(c, CLAUSE_TEXT) for c in candidates
            if c.marker and (c.keyword or "." not in c.marker) and (c.known or c.heading_block)][:80]


def question(group: list[Candidate], position: int) -> dict[str, Any]:
    if len(group) == 1:
        return {"type": "choice",
                "instructions": f"¿La línea state.lines[{position}] (la de index {group[0].index} en state.outline) abre un "
                                "apartado del documento, que alguien puede pedir por su número o su título, o es contenido "
                                "dentro de un apartado? Aplica state.convention.",
                "criteria": CRITERIA}
    refs = ", ".join(str(c.index) for c in group)
    return {"type": "choice",
            "instructions": f"Las líneas state.lines[{position}].siblings (las de index {refs} en state.outline) comparten "
                            "numeración padre y nivel. ¿Abren subapartados del documento, que alguien puede pedir por su "
                            "número aunque sean párrafos, o son una lista que empieza de nuevo dentro del apartado padre? "
                            "Decide para todas a la vez. Aplica state.convention.",
            "criteria": CRITERIA}


class TypesafeHeadingJudge:
    """Jev over HTTP (`POST {base_url}/systemone`), as the TypeSafe provider of the AI SDK sends it: a JSON state and
    choice questions; the answers carry the choice and its probabilities."""

    name = "typesafe"

    def __init__(self, api_key: str, model: str = TYPESAFE_MODEL, base_url: str = TYPESAFE_URL,
                 timeout: float = 90.0, workers: int = 4, session: Any = None,
                 pdf_path: Optional[str] = None, pages: Optional[list[dict[str, Any]]] = None,
                 images: bool = False):
        if not api_key:
            raise ValueError("The Typesafe judge needs an API key")
        self._api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.workers = max(1, workers)
        self._session = session
        # With the PDF and its stored pages, each asked line goes with a crop of its page: size, weight, indent
        # and position are what a reader sees and the text alone does not carry.
        self.pdf_path = pdf_path
        self.pages = pages
        self.images = images and pdf_path is not None and pages is not None
        self.usage = {"input_tokens": 0, "output_tokens": 0, "requests": 0, "failed": 0, "images": 0}

    def decide(self, document: dict[str, Any], candidates: list[Candidate],
               ask: list[Candidate]) -> dict[int, tuple[str, float]]:
        if not ask:
            return {}
        shared = {"convention": CONVENTION, "document": document, "clauses": clauses(candidates)}
        # with page crops a request carries far more tokens: fewer lines per request
        parts = stretches(sibling_groups(ask), IMAGE_STRETCH_LINES if self.images else STRETCH_LINES)
        decided: dict[int, tuple[str, float]] = {}
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for result in pool.map(lambda stretch: self._evaluate(candidates, stretch, shared), parts):
                decided.update(result)
        return decided

    def _evaluate(self, candidates: list[Candidate], stretch: list[list[Candidate]],
                  shared: dict[str, Any]) -> dict[int, tuple[str, float]]:
        lines = [line_state(g[0]) if len(g) == 1 else {"siblings": [line_state(c) for c in g]} for g in stretch]
        images: dict[str, bytes] = {}
        if self.images:
            from .crops import line_crop
            for i, group in enumerate(stretch):
                for k, c in enumerate(group):
                    png = line_crop(self.pdf_path, self.pages, c)
                    if png is None:
                        continue
                    ref = f"[linea-{c.index}]"
                    images[ref] = png
                    target = lines[i] if len(group) == 1 else lines[i]["siblings"][k]
                    target["imagen"] = ref
        state = {**shared, "outline": outline_window(candidates, stretch), "lines": lines}
        if images:
            state["imagenes"] = ("Cada línea preguntada lleva en `imagen` un recorte de su página: la línea marcada con un "
                                 "recuadro rojo y su entorno. Úsalo para ver tamaño, negrita, sangría y posición.")
        questions = {f"line_{i}": question(g, i) for i, g in enumerate(stretch)}
        try:
            answers = self._post({"model": self.model, "state": state, "questions": questions}, images)
        except Exception as exc:  # a failed request: halves, then the rules
            self.usage["failed"] += 1
            logger.warning("headings judge request failed (%d lines): %s", sum(len(g) for g in stretch), exc)
            if len(stretch) < 2:
                return {}
            half = len(stretch) // 2
            return {**self._evaluate(candidates, stretch[:half], shared),
                    **self._evaluate(candidates, stretch[half:], shared)}
        decided: dict[int, tuple[str, float]] = {}
        for i, group in enumerate(stretch):
            answer = answers.get(f"line_{i}") or {}
            choice = answer.get("choice")
            if answer.get("type") != "choice" or choice not in CRITERIA:
                continue
            probability = float((answer.get("probabilities") or {}).get(choice, 0.0))
            for c in group:
                decided[c.index] = (choice, probability)
        return decided

    def _post(self, body: dict[str, Any], images: Optional[dict[str, bytes]] = None) -> dict[str, Any]:
        session = self._session
        if session is None:
            import requests
            session = requests
        headers = {"Authorization": f"Bearer {self._api_key}"}
        if images:
            # TypeSafe's multipart form: the request as JSON plus one part per image, named in `assets`.
            import json as _json
            parts = {ref: f"image-{n}" for n, ref in enumerate(images)}
            body = {**body, "assets": {ref: {"type": "image", "part": part} for ref, part in parts.items()}}
            files = {"request": ("request.json", _json.dumps(body, ensure_ascii=False).encode(), "application/json")}
            files.update({part: (f"{part}.png", images[ref], "image/png") for ref, part in parts.items()})
            self.usage["images"] += len(images)
            response = session.post(f"{self.base_url}/systemone", files=files, timeout=self.timeout,
                                    headers={**headers, "TypeSafe-Danger-Zone-Images": "true"})
        else:
            response = session.post(f"{self.base_url}/systemone", json=body, timeout=self.timeout, headers=headers)
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage") or {}
        self.usage["requests"] += 1
        self.usage["input_tokens"] += int(usage.get("input_tokens") or 0)
        self.usage["output_tokens"] += int(usage.get("output_tokens") or 0)
        return payload.get("answers") or {}
