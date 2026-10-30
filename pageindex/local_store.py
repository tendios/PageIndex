"""Compatibility alias: the on-disk store moved to pageindex.stores.file."""
from .stores.file import (DocStore, FileDocumentStore,  # noqa: F401
                          _read_json, _write_json_atomic)
