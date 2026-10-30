"""Pluggable document stores for local mode.

    PageIndexClient(index={"store": {"type": "s3", "bucket": "my-bucket"}})
    PageIndexClient(index={"store": SqlDocumentStore("postgresql+psycopg://…")})

Every store keeps the same JSON documents under the same ids, so an index
moves between them with ``copy(source, target)``.
"""
from __future__ import annotations

from typing import Any, Mapping, Union

from .base import DocumentStore
from .file import FileDocumentStore

__all__ = ["DocumentStore", "FileDocumentStore", "S3DocumentStore",
           "SqlDocumentStore", "open_store", "copy"]

_LAZY = {"S3DocumentStore": ".s3", "SqlDocumentStore": ".sql"}
_TYPES = {"file": "FileDocumentStore", "s3": "S3DocumentStore",
          "sql": "SqlDocumentStore"}


def __getattr__(name):
    # boto3 / SQLAlchemy load only when their store is used.
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name], __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


StoreSpec = Union[DocumentStore, Mapping[str, Any]]


def open_store(spec: StoreSpec) -> DocumentStore:
    """A store from an instance (returned as-is) or a dict:
    ``{"type": "file", "path": …}``, ``{"type": "s3", "bucket": …,
    "prefix": …, "endpoint_url": …}`` or ``{"type": "sql", "url": …}`` —
    the keys besides "type" are the store's constructor arguments."""
    if isinstance(spec, DocumentStore):
        return spec
    if not isinstance(spec, Mapping):
        raise TypeError("store must be a DocumentStore or a dict with a "
                        f'"type", got {type(spec).__name__}.')
    options = dict(spec)
    kind = options.pop("type", None)
    if kind not in _TYPES:
        raise ValueError(f'store "type" must be "file", "s3" or "sql", '
                         f"got {kind!r}.")
    if kind == "file" and "path" in options:
        options["storage_dir"] = options.pop("path")
    cls = globals().get(_TYPES[kind]) or __getattr__(_TYPES[kind])
    import inspect
    try:
        inspect.signature(cls).bind(**options)
    except TypeError as e:
        raise ValueError(f'store {{"type": "{kind}"}}: {e}') from None
    return cls(**options)


def copy(source: StoreSpec, target: StoreSpec, *,
         overwrite: bool = False) -> list[str]:
    """Copy every document from ``source`` to ``target`` (stores or
    dicts, as ``open_store`` takes), ids and JSON unchanged. Documents
    already in the target are skipped unless ``overwrite``. Returns the
    copied ids."""
    source, target = open_store(source), open_store(target)
    copied = []
    for meta in source.list_metas():
        doc_id = meta["id"]
        if not overwrite and target.get_meta(doc_id) is not None:
            continue
        tree, pages = source.get_tree(doc_id), source.get_pages(doc_id)
        if tree is None or pages is None:  # deleted meanwhile
            continue
        target.save_document(doc_id, meta, tree, pages)
        copied.append(doc_id)
    return copied
