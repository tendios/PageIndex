"""The document store interface behind PageIndexClient's local mode."""
from __future__ import annotations

import os
from abc import ABC, abstractmethod
from contextlib import AbstractContextManager


def _is_safe_id(value: str) -> bool:
    return (
        isinstance(value, str)
        and value not in ("", ".", "..")
        and os.path.basename(value) == value
        and "\\" not in value
    )


def _is_valid_meta(meta, doc_id: str) -> bool:
    if not isinstance(meta, dict) or meta.get("id") != doc_id:
        return False
    page_num = meta.get("pageNum")
    return (
        isinstance(meta.get("name"), str)
        and (meta.get("description") is None
             or isinstance(meta.get("description"), str))
        and isinstance(meta.get("status"), str)
        and isinstance(meta.get("createdAt"), str)
        and isinstance(page_num, int)
        and not isinstance(page_num, bool)
        and page_num >= 0
        and (meta.get("folderId") is None
             or isinstance(meta.get("folderId"), str))
        and (meta.get("metadata") is None
             or isinstance(meta.get("metadata"), dict))
        and (meta.get("mode") is None or isinstance(meta.get("mode"), str))
    )


class DocumentStore(ABC):
    """Where local mode keeps indexed documents.

    A document is three JSON values under one ``doc_id``: ``meta`` (the
    ``doc.json`` dict: id, name, description, status, createdAt, pageNum,
    folderId, metadata, mode), ``tree`` (the node list) and ``pages``
    (``[{"page_index", "markdown"}]``). Every store keeps them as-is, so
    a document copies between stores unchanged (see ``stores.copy``).
    """

    @abstractmethod
    def save_document(self, doc_id: str, meta: dict, tree: list,
                      pages: list) -> None:
        """Store a document. Atomic per document: its meta is written last
        and marks it, so a document whose save failed midway is never
        listed nor read. Raises ValueError on an invalid id."""

    @abstractmethod
    def get_meta(self, doc_id: str) -> dict | None:
        """The document's meta, or None when absent or unreadable."""

    @abstractmethod
    def get_tree(self, doc_id: str) -> list | None:
        """The document's tree, or None when absent."""

    @abstractmethod
    def get_pages(self, doc_id: str) -> list | None:
        """The document's pages, or None when absent."""

    @abstractmethod
    def list_metas(self) -> list[dict]:
        """The metas of every stored document, in any order — the client
        sorts newest first and pages with limit/offset itself."""

    @abstractmethod
    def delete_document(self, doc_id: str) -> bool:
        """Remove a document; True when it existed."""

    @abstractmethod
    def lock(self) -> AbstractContextManager:
        """Mutex for check-then-write sequences on this store (the client
        lists names, picks a free one and saves under it). It only guards
        those sequences against other writers of the same store; plain
        reads and writes never need it."""

    # Optional: the library topic index (PageIndexClient.build_library_topics).
    # A store without it simply has none.
    def get_library_index(self) -> dict | None:
        """The stored library index, or None."""
        return None

    def save_library_index(self, index: dict) -> None:
        """Store (replace) the library index."""
        raise NotImplementedError(
            f"{type(self).__name__} does not store a library index.")
