"""Local implementation of the PageIndex SDK surface."""
from __future__ import annotations

import json
import logging
import multiprocessing
import os
import re
import uuid
from time import perf_counter
from datetime import datetime, timezone
from typing import Any

from .errors import PageIndexAPIError
from .stores.base import DocumentStore
from .stores.file import FileDocumentStore
from .naming import sanitize_filename, truncate_filename
from .utils import LLMRetriesExhausted, count_tokens, run_off_loop

logger = logging.getLogger(__name__)

_SURROGATES = re.compile("[\ud800-\udfff]")
# A caller's doc_id is a path segment and an object key in every store.
_DOC_ID = re.compile(r"[A-Za-z0-9._-]{1,128}")


def _scrub_surrogates(text: str) -> str:
    """Lone surrogates (surrogateescape'd names, PyPDF2's surrogatepass
    decodes) cannot encode to UTF-8; replace with U+FFFD."""
    return _SURROGATES.sub("\ufffd", text)


def _now_iso() -> str:
    """Naive UTC, millisecond precision."""
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    return now.replace(microsecond=now.microsecond // 1000 * 1000).isoformat()




class LocalAPI:
    """Backs PageIndexClient's local mode. One instance per client."""

    def __init__(self, storage_path: str, model: str, summary_model: str,
                 index_backend: dict | None = None,
                 summary_max_words: int | None = None,
                 summary_concurrency: int | None = None,
                 use_embedded_toc: bool = True,
                 optimize: str = "full",
                 summary_reasoning_effort: str | None = None,
                 store: DocumentStore | None = None,
                 pdf_parser: str | dict | None = None,
                 heading_judge: str | dict | None = None,
                 summary_language: str | None = None):
        self._store = store if store is not None else FileDocumentStore(storage_path)
        self._model = model
        self._summary_model = summary_model
        self._index_backend = index_backend
        self._summary_max_words = summary_max_words
        self._summary_reasoning_effort = summary_reasoning_effort
        self._summary_concurrency = summary_concurrency
        self._use_embedded_toc = use_embedded_toc
        self._optimize = optimize
        from .pdf_extraction import parser_config
        self._pdf_parser = parser_config(pdf_parser)
        # mode="headings": who decides the lines no rule settles: "typesafe" (Jev), "none" (the rules) or
        # "auto" (the default: Jev when TYPESAFE_AI_API_KEY is set).
        self._heading_judge = heading_judge
        # The language of summaries and description: None for the document's own, or a language name to force.
        self._summary_language = summary_language
        from .utils import ConfigLoader
        self._config_loader = ConfigLoader()

    def _with_backend(self, usage, func, *args):
        """Scope the indexing lane's connection overrides and the `usage`
        tally around one operation — runs inside whatever thread
        run_off_loop picked."""
        from .utils import _llm_backend, _llm_usage, _summary_language
        token = _llm_backend.set(self._index_backend)
        usage_token = _llm_usage.set(usage)
        language_token = _summary_language.set(self._summary_language)
        try:
            return func(*args)
        finally:
            _summary_language.reset(language_token)
            _llm_usage.reset(usage_token)
            _llm_backend.reset(token)

    # ── indexing ──

    def submit_document(
        self,
        file_path: str,
        mode: str | None = None,
        beta_headers: list[str] | None = None,
        folder_id: str | None = None,
        metadata: dict | None = None,
        doc_id: str | None = None,
        if_exists: str | None = None,
    ) -> dict[str, Any]:
        if getattr(multiprocessing.current_process(), "_inheriting", False):
            raise PageIndexAPIError(
                "Failed to submit document: called again while a spawned worker "
                "process was importing your script. Put your top-level code under "
                "if __name__ == '__main__': so worker processes do not re-run it."
            )
        if beta_headers is not None:
            raise PageIndexAPIError(
                "Failed to submit document: beta_headers is not supported in local mode."
            )
        if folder_id is not None:
            raise PageIndexAPIError(
                "Failed to submit document: folders are not supported in local mode."
            )
        if metadata is not None:
            if not isinstance(metadata, dict):
                raise PageIndexAPIError(
                    "Failed to submit document: metadata must be a dict."
                )
            try:
                json.dumps(metadata, allow_nan=False)
            except (TypeError, ValueError) as e:
                raise PageIndexAPIError(
                    f"Failed to submit document: metadata must be valid JSON. {e}"
                ) from e
        if doc_id is not None and not (
                isinstance(doc_id, str) and _DOC_ID.fullmatch(doc_id)
                and doc_id not in (".", "..")):
            raise PageIndexAPIError(
                f"Failed to submit document: invalid doc_id {doc_id!r} — 1 to "
                "128 letters, digits, '-', '_' or '.'.", status_code=400)
        if if_exists not in (None, "error", "replace"):
            raise PageIndexAPIError(
                "Failed to submit document: if_exists must be \"error\" or "
                f"\"replace\", got {if_exists!r}.", status_code=400)
        if if_exists is not None and doc_id is None:
            raise PageIndexAPIError(
                "Failed to submit document: if_exists applies to a given "
                "doc_id only.", status_code=400)
        replace = if_exists == "replace"
        if doc_id is not None and not replace:
            self._check_doc_id_free(doc_id)  # before indexing; rechecked at save
        if mode not in (None, "standard", "flash", "headings"):
            raise PageIndexAPIError(
                f"Failed to submit document: unknown local processing mode {mode!r}. "
                "Supported: 'headings' (the default with pdf_parser='liteparse'), "
                "'flash' (the default otherwise) or 'standard'."
            )
        if mode is None:
            mode = self.default_mode
        if mode == "standard" and (self._summary_max_words is not None
                                   or self._summary_concurrency is not None
                                   or self._summary_reasoning_effort is not None):
            raise PageIndexAPIError(
                "Failed to submit document: summary_max_words, "
                "summary_concurrency and summary_reasoning_effort are "
                "flash-only; mode='standard' does not support them.")
        file_path = os.path.abspath(os.path.expanduser(str(file_path)))
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"No such file: {file_path}")
        doc_name = sanitize_filename(os.path.basename(file_path))
        if not doc_name.lower().endswith(".pdf"):
            raise PageIndexAPIError(
                "Failed to submit document: only PDF files are supported in local mode."
            )

        extraction = None
        started = perf_counter()
        try:
            if self._pdf_parser["name"] == "liteparse":
                from .pdf_extraction import extract_liteparse
                extraction = extract_liteparse(file_path, self._pdf_parser)
                page_texts = extraction.texts
            else:
                page_texts = self._extract_page_texts(file_path)
        except PageIndexAPIError:
            raise
        except Exception as e:
            raise PageIndexAPIError(
                f"Failed to submit document: could not read PDF: {e}"
            ) from e
        if not any(text.strip() for text in page_texts):
            raise PageIndexAPIError(
                "Failed to submit document: PDF has no content. All pages are blank."
            )
        self._unique_doc_name(doc_name, doc_id if replace else None)
        # Seconds per stage, returned with the result: extract, then the mode's own stages, then save.
        timings = {"extract": perf_counter() - started}
        if extraction is not None:
            extraction.timings = timings

        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                 "prompt_tokens_details": {"cached_tokens": 0,
                                           "cache_write_tokens": 0},
                 "completion_tokens_details": {"reasoning_tokens": 0}}
        try:
            if mode == "flash":
                structure, description = run_off_loop(
                    self._with_backend, usage, self._index_flash, file_path,
                    *([extraction] if extraction is not None else [])
                )
            elif mode == "headings":
                structure, description = run_off_loop(
                    self._with_backend, usage, self._index_headings, file_path,
                    extraction, page_texts, doc_name
                )
            else:
                structure, description = run_off_loop(
                    self._with_backend, usage, self._index_standard, file_path,
                    page_texts
                )
        except PageIndexAPIError:
            raise
        except LLMRetriesExhausted as e:
            # with a status its message names the cause; the chain stays on __context__
            raise PageIndexAPIError(f"Failed to submit document: {e}") from (
                None if e.status_code else e)
        except Exception as e:
            raise PageIndexAPIError(f"Failed to submit document: {e}") from e
        self._check_page_bounds(structure, len(page_texts))
        tick = perf_counter()

        if doc_id is None:
            doc_id = "pi-" + uuid.uuid4().hex
        pages = extraction.pages if extraction is not None else [
            {"page_index": i + 1, "markdown": text} for i, text in enumerate(page_texts)]
        from .utils import remove_fields
        # Check-then-write under the store lock; the early pre-check above
        # is advisory only.
        with self._store.lock():
            if not replace:
                self._check_doc_id_free(doc_id)
            meta = {
                "id": doc_id,
                "name": self._unique_doc_name(
                    doc_name, doc_id if replace else None),
                "description": description,
                "status": "completed",
                "createdAt": _now_iso(),
                "pageNum": len(page_texts),
                "folderId": None,
                "metadata": metadata,
                "mode": mode,
            }
            if extraction is not None:
                meta["pdf_extraction"] = extraction.metadata
            self._store.save_document(
                doc_id, meta, remove_fields(structure, fields=["text"]), pages)
        timings["save"] = perf_counter() - tick
        timings["total"] = perf_counter() - started
        return {"doc_id": doc_id, "name": meta["name"], "usage": usage,
                "timings": {stage: round(seconds, 2) for stage, seconds in timings.items()}}

    def _check_doc_id_free(self, doc_id: str) -> None:
        if self._store.get_meta(doc_id) is not None:
            raise PageIndexAPIError(
                f"Failed to submit document: doc_id {doc_id!r} already "
                'exists — pass if_exists="replace" to reindex it.',
                status_code=409)

    def _unique_doc_name(self, name: str, replacing: str | None = None) -> str:
        """Mirror the cloud upload: a taken name gets _1.._99 appended,
        beyond that the submit is rejected. The document being replaced
        does not hold its own name."""
        taken = {meta.get("name") for meta in self._store.list_metas()
                 if replacing is None or meta.get("id") != replacing}
        if name not in taken:
            return name
        for num in range(1, 100):
            candidate = truncate_filename(name, suffix=f"_{num}")
            if candidate not in taken:
                return candidate
        raise PageIndexAPIError(
            "Failed to submit document: Too many files with similar names. "
            "Please use a different file name."
        )

    def build_library_topics(self) -> dict[str, Any]:
        from .library import build_topic_index, document_root
        from .utils import llm_completion
        metas = self._store.list_metas()
        entries = [{"id": m.get("id"), "name": m.get("name"),
                    "pageNum": m.get("pageNum"),
                    "description": m.get("description"),
                    "metadata": m.get("metadata")} for m in metas
                   if m.get("status") == "completed"]
        roots = [document_root(self, entry) for entry in entries]
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
                 "prompt_tokens_details": {"cached_tokens": 0,
                                           "cache_write_tokens": 0},
                 "completion_tokens_details": {"reasoning_tokens": 0}}

        def complete(prompt: str) -> str:
            return run_off_loop(self._with_backend, usage, llm_completion,
                                self._model, prompt)

        try:
            index = build_topic_index(
                roots, complete, model=self._model,
                created_at={m.get("id"): m.get("createdAt") for m in metas})
        except LLMRetriesExhausted as e:
            raise PageIndexAPIError(
                f"Failed to build library topics: {e}") from e
        self._store.save_library_index(index)
        return {"topics": len(index["topics"]), "documents": len(roots),
                "usage": usage}

    @staticmethod
    def _check_page_bounds(structure: list, page_count: int) -> None:
        """The tree (pdfium) and stored pages (PyPDF2) come from different
        parsers; a span outside 1..page_count IndexErrors every later read."""
        stack = list(structure)
        while stack:
            node = stack.pop()
            start, end = node.get("start_index"), node.get("end_index")
            if (start is not None and end is not None
                    and not (1 <= start and end <= page_count)):
                raise PageIndexAPIError(
                    f"Failed to submit document: the extracted structure "
                    f"references pages {start}-{end} outside the PDF's "
                    f"{page_count} readable pages."
                )
            stack.extend(node.get("nodes") or [])

    @staticmethod
    def _extract_page_texts(file_path: str) -> list[str]:
        import PyPDF2
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            # PyPDF2 decodes broken ToUnicode maps with surrogatepass; lone
            # surrogates would crash every utf-8 JSON save downstream.
            return [_scrub_surrogates(page.extract_text() or "")
                    for page in reader.pages]

    def _index_standard(self, file_path: str, page_texts: list[str]) -> tuple[list, str | None]:
        from .page_index_classic import page_index_main
        page_list = [(text, count_tokens(text, model=self._model)) for text in page_texts]
        opt = self._config_loader.load({
            "model": self._model,
            "summary_model": self._summary_model,
            "if_add_node_id": "yes",
            "if_add_node_summary": "yes",
            "if_add_node_text": "no",
            "if_add_doc_description": "yes",
        })
        result = page_index_main(file_path, opt, logger=logger, page_list=page_list)
        structure = result.get("structure") or []
        if not structure:
            raise PageIndexAPIError(
                "Failed to submit document: standard indexing produced no structure."
            )
        return structure, result.get("doc_description")

    def _index_flash(self, file_path: str, extraction=None) -> tuple[list, str | None]:
        from .flash import page_index_flash
        from .flash.api import flash_rejection_reason
        from .utils import (create_clean_structure_for_description,
                            generate_doc_description, write_node_id)
        result = page_index_flash(file_path, summary=True,
                                  summary_model=self._summary_model,
                                  optimize=False if self._optimize == "off" else self._optimize,
                                  optimize_model=self._summary_model,
                                  summary_concurrency=self._summary_concurrency,
                                  summary_max_words=self._summary_max_words,
                                  summary_reasoning_effort=self._summary_reasoning_effort,
                                  use_embedded_toc=self._use_embedded_toc,
                                  **({"page_texts": extraction.texts}
                                     if extraction is not None else {}))
        structure = result.get("structure", [])
        reason = flash_rejection_reason(result)
        if reason and extraction is not None and result.get("toc_source") == "pages":
            # OCR gives us text but cannot recover the native font geometry
            # Flash uses for headings. Its large flat-tree refusal occurs
            # before any LLM calls; build a proper tree with the existing
            # standard backend, using exactly the same extracted pages.
            if any(value is not None for value in (self._summary_max_words,
                                                   self._summary_concurrency,
                                                   self._summary_reasoning_effort)):
                raise PageIndexAPIError(
                    "Failed to submit document: OCR recovered text but Flash found no "
                    "hierarchy. Standard fallback cannot honor flash-only summary "
                    "options; remove those options and use mode='standard'.")
            logger.info("LiteParse: no usable layout hierarchy; using standard indexing")
            extraction.metadata["structure_backend"] = "standard_fallback"
            return self._index_standard(file_path, extraction.texts)
        if reason:
            raise PageIndexAPIError(f"Failed to submit document: {reason}")
        write_node_id(structure)
        description = generate_doc_description(
            create_clean_structure_for_description(structure, top_level_only=True),
            model=self._summary_model,
        )
        return structure, description

    @property
    def default_mode(self) -> str:
        """The mode of a submit that names none: ``headings`` when LiteParse reads the PDF (its blocks are the
        headings), ``flash`` with the default reader, as upstream."""
        return "headings" if self._pdf_parser["name"] == "liteparse" else "flash"

    def _heading_judge_instance(self, pdf_path: str | None = None, pages: list | None = None):
        """The judge of mode="headings": Typesafe's Jev with the configured key, or none, which leaves the
        undecided lines to the rules. Unset or ``"auto"``: Jev when TYPESAFE_AI_API_KEY is in the environment,
        the rules otherwise."""
        conf = self._heading_judge
        if conf is None or conf == "":
            conf = "auto"
        if conf == "none":
            return None
        if isinstance(conf, str):
            conf = {"name": conf}
        name = (conf.get("name") or "").strip().lower()
        if name == "auto":  # Jev when a key is at hand (the dict's or the environment's), the rules otherwise
            if not (conf.get("api_key") or os.environ.get("TYPESAFE_AI_API_KEY", "")):
                return None
            conf, name = {**conf, "name": "typesafe"}, "typesafe"
        if name != "typesafe":
            raise PageIndexAPIError(
                f"Failed to submit document: unknown heading_judge {name!r}; use \"auto\", \"typesafe\" or \"none\".")
        api_key = conf.get("api_key") or os.environ.get("TYPESAFE_AI_API_KEY", "")
        if not api_key:
            raise PageIndexAPIError(
                "Failed to submit document: heading_judge \"typesafe\" needs an api_key, or TYPESAFE_AI_API_KEY "
                "in the environment.")
        from .headings import TYPESAFE_MODEL, TYPESAFE_URL, TypesafeHeadingJudge
        # "images": True sends each doubtful line with a crop of its page (size, weight, indent, position).
        return TypesafeHeadingJudge(api_key=api_key, model=conf.get("model") or TYPESAFE_MODEL,
                                    base_url=conf.get("base_url") or TYPESAFE_URL,
                                    timeout=float(conf.get("timeout") or 90), workers=int(conf.get("workers") or 8),
                                    pdf_path=pdf_path, pages=pages, images=bool(conf.get("images")),
                                    verify=bool(conf.get("verify", True)))

    def _index_headings(self, file_path: str, extraction, page_texts: list[str],
                        doc_name: str) -> tuple[list, str | None]:
        """The tree from the document's own headings (see ``pageindex.headings``), then the flash summaries and
        description. Needs LiteParse: the headings are its blocks."""
        import asyncio
        from .flash.api import _page_nodes, _summarize
        from .headings import build_headings_structure
        from .utils import (SUMMARY_MAX_WORDS, create_clean_structure_for_description,
                            default_summary_reasoning_effort, generate_doc_description,
                            learn_model_if_effort_dropped, write_node_id)
        if extraction is None:
            raise PageIndexAPIError(
                "Failed to submit document: mode='headings' reads the headings LiteParse finds; configure "
                "pdf_parser='liteparse'.")
        timings = getattr(extraction, "timings", {})  # stubs in tests may lack it
        tick = perf_counter()
        layout_structure = None
        try:
            # The layout titles flash detects, without its model passes: the same entry tests and callers stub.
            from .flash import page_index_flash
            detected = page_index_flash(file_path, summary=False, optimize=False,
                                        use_embedded_toc=self._use_embedded_toc)
            if detected.get("toc_source") not in ("pages", "unreadable"):
                layout_structure = detected.get("structure")
        except Exception as exc:  # scanned or odd PDFs: the headings alone decide
            logger.info("headings: layout detection unavailable (%s)", exc)
        timings["layout"], tick = perf_counter() - tick, perf_counter()
        structure, report = build_headings_structure(
            extraction.pages, page_texts, layout_structure,
            judge=self._heading_judge_instance(file_path, extraction.pages),
            document={"name": doc_name, "pages": len(page_texts)})
        timings["headings"], tick = perf_counter() - tick, perf_counter()  # the rules and Jev, its final review included
        backend = "headings"
        if not structure:
            structure, backend = _page_nodes(page_texts), "headings_pages"
        extraction.metadata["structure_backend"] = backend
        extraction.metadata["headings"] = {key: value for key, value in report.items() if key != "decisions"}
        write_node_id(structure)
        summary_model = self._summary_model or self._model
        from .integrations.claude_agent_sdk import is_claude_code
        if not is_claude_code(summary_model):
            learn_model_if_effort_dropped(summary_model)
        effort = (self._summary_reasoning_effort or "").strip() or None
        defaulted = effort is None
        if defaulted:
            effort = default_summary_reasoning_effort(summary_model)
        asyncio.run(_summarize(structure, [(text, 0) for text in page_texts], summary_model,
                               concurrency=self._summary_concurrency,
                               max_words=self._summary_max_words or SUMMARY_MAX_WORDS,
                               reasoning_effort=effort, drop_refused_effort=defaulted))
        timings["summaries"], tick = perf_counter() - tick, perf_counter()
        description = generate_doc_description(
            create_clean_structure_for_description(structure, top_level_only=True), model=summary_model)
        timings["description"] = perf_counter() - tick
        return structure, description

    # ── tree / ocr ──

    def _load_tree_with_text(self, doc_id: str, error_prefix: str) -> list:
        """Each node's own text (see utils.own_pages)."""
        from .utils import add_node_text
        structure = self._require_data(
            self._store.get_tree(doc_id), error_prefix)
        pages = self._require_pages(doc_id, error_prefix)
        pdf_pages = [(p.get("markdown", ""), 0) for p in pages]
        add_node_text(structure, pdf_pages)
        return structure

    def get_tree(self, doc_id: str, node_summary: bool = False,
                 include_text: bool = True) -> dict[str, Any]:
        meta = self._require_doc(doc_id, "Failed to get tree result")
        if include_text:
            structure = self._load_tree_with_text(doc_id, "Failed to get tree result")
        else:
            structure = self._require_data(
                self._store.get_tree(doc_id), "Failed to get tree result")
        result = [_format_tree_node(node, node_summary) for node in structure]
        return self._completed_envelope(doc_id, result, meta)

    def get_ocr(self, doc_id: str, format: str = "page") -> dict[str, Any]:
        if format not in ["page", "node", "raw"]:
            raise ValueError("Format parameter must be 'page', 'node', or 'raw'")
        meta = self._require_doc(doc_id, "Failed to get OCR result")
        if format == "node":
            result: Any = []
            def _walk(nodes, level):
                for node in nodes:
                    result.append({
                        "title": node.get("title", ""),
                        "level": level,
                        "page_index": node.get("start_index"),
                        "text": node.get("text", ""),
                    })
                    _walk(node.get("nodes") or [], level + 1)
            _walk(self._load_tree_with_text(doc_id,
                                           "Failed to get OCR result"), 1)
        else:
            pages = self._require_pages(doc_id, "Failed to get OCR result")
            if format == "page":
                result = pages
            else:  # raw
                result = "\n\n".join(p.get("markdown", "") for p in pages)
        return self._completed_envelope(doc_id, result, meta)

    def get_block(self, doc_id: str, block_id: str) -> dict[str, Any]:
        self._require_doc(doc_id, "Failed to get block")
        pages = self._require_pages(doc_id, "Failed to get block")
        if not any("blocks" in page for page in pages):
            raise PageIndexAPIError(
                "get_block is cloud-only for legacy documents; reindex locally "
                "with pdf_parser='liteparse' to retain layout blocks.")
        for page in pages:
            for block in page.get("blocks") or []:
                if block.get("block_id") == block_id:
                    return {"doc_id": doc_id, **block}
        raise PageIndexAPIError(f"Block {block_id!r} not found", status_code=404)

    @staticmethod
    def _require_data(data, error_prefix: str):
        if data is None:
            raise PageIndexAPIError(f"{error_prefix}: stored document data is unreadable.")
        return data

    def _require_pages(self, doc_id: str, error_prefix: str) -> list:
        pages = self._require_data(self._store.get_pages(doc_id), error_prefix)
        if not pages:
            raise PageIndexAPIError(
                f"{error_prefix}: stored document has no page content.")
        return pages

    def _completed_envelope(self, doc_id: str, result, meta: dict) -> dict[str, Any]:
        return {
            "doc_id": doc_id,
            "status": "completed",
            "retrieval_ready": True,
            "result": result,
            "metadata": meta.get("metadata"),
            "features": ({"pdf_extraction": meta["pdf_extraction"]}
                         if "pdf_extraction" in meta else {}),
        }

    # ── document management ──

    def _require_doc(self, doc_id: str, error_prefix: str) -> dict:
        meta = self._store.get_meta(doc_id)
        if meta is None:
            raise PageIndexAPIError(f"{error_prefix}: Document not found.")
        return meta

    def get_document(self, doc_id: str) -> dict[str, Any]:
        meta = self._store.get_meta(doc_id)
        if meta is None:
            raise PageIndexAPIError("Failed to get document metadata: Document not found")
        return {key: meta.get(key) for key in
                ("id", "name", "description", "status", "createdAt", "pageNum",
                 "folderId", "metadata")}

    def delete_document(self, doc_id: str) -> dict[str, Any]:
        if not self._store.delete_document(doc_id):
            raise PageIndexAPIError("Failed to delete document: Document not found.")
        return {"message": "Document deleted successfully."}

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        folder_id: str | None = None,
        name: str | None = None,
        recursive: bool = False,
    ) -> dict[str, Any]:
        if limit < 1 or limit > 10000:
            raise ValueError("limit must be between 1 and 10000")
        if offset < 0:
            raise ValueError("offset must be non-negative")
        if folder_id is not None and folder_id not in ("", "root"):
            raise PageIndexAPIError(
                "Failed to list documents: folders are not supported in local mode."
            )
        metas = sorted(self._store.list_metas(), key=lambda m: m.get("id") or "")
        metas.sort(key=lambda m: m.get("createdAt") or "", reverse=True)
        if name is not None:
            metas = [m for m in metas if m.get("name") == name]
        documents = [{
            "id": m.get("id"),
            "name": m.get("name"),
            "description": m.get("description"),
            "status": m.get("status"),
            "createdAt": m.get("createdAt"),
            "pageNum": m.get("pageNum", 0),
            "folderId": None,
            "path": None,
            "metadata": m.get("metadata"),
            "features": {},
        } for m in metas[offset:offset + limit]]
        return {
            "documents": documents,
            "total": len(metas),
            "limit": limit,
            "offset": offset,
        }


def _format_tree_node(node: dict, node_summary: bool) -> dict:
    children = node.get("nodes") or []
    out = {
        "title": node.get("title", ""),
        "node_id": node.get("node_id"),
        "start_index": node.get("start_index"),
        "end_index": node.get("end_index"),
    }
    if node.get("key_items"):
        out["key_items"] = node["key_items"]
    if node_summary:
        summary = node.get("summary")
        if summary is not None:
            out["summary"] = summary
    if "text" in node:
        out["text"] = node["text"]
    if children:
        out["nodes"] = [_format_tree_node(child, node_summary) for child in children]
    return out
