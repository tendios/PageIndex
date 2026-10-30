"""SQL document store (SQLAlchemy Core): Postgres, SQLite, …"""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager

from .base import DocumentStore, _is_safe_id, _is_valid_meta

try:
    import sqlalchemy as sa
except ImportError:  # the [sql] extra
    sa = None


def _scrub(text: str) -> str:
    # A lone surrogate is replaced, as the file store does.
    return text.encode("utf-8", "replace").decode("utf-8")


def _dumps(data) -> str:
    return _scrub(json.dumps(data, ensure_ascii=False))


class SqlDocumentStore(DocumentStore):
    """One row per document in ``pageindex_documents`` (doc_id, name,
    meta, tree, pages as JSON, created_at), created on first use; the
    library index is a row of ``pageindex_library``.

    ``url`` is a SQLAlchemy URL (``postgresql+psycopg://…``,
    ``sqlite:///index.db``) or an Engine. ``lock()`` is a transaction
    that first updates the single row of ``pageindex_store_lock`` — a
    write lock every backend serializes on — and the store's own calls
    made inside it join that transaction, so the name check and the
    save commit together.
    """

    def __init__(self, url, *, table: str = "pageindex_documents"):
        if sa is None:
            raise ImportError(
                "SqlDocumentStore needs SQLAlchemy — "
                'pip install "pageindex[sql]" (plus your database driver, '
                "e.g. psycopg).")
        if isinstance(url, sa.engine.Engine):
            self._engine = url
        else:
            options = {"json_serializer": _dumps}
            parsed = sa.engine.make_url(url)
            if (parsed.get_backend_name() == "sqlite"
                    and parsed.database in (None, "", ":memory:")):
                # One shared connection, or each thread sees its own empty
                # database. In-memory SQLite is for tests.
                options.update(poolclass=sa.pool.StaticPool,
                               connect_args={"check_same_thread": False})
            self._engine = sa.create_engine(url, **options)
        metadata = sa.MetaData()
        self._docs = sa.Table(
            table, metadata,
            sa.Column("doc_id", sa.String(255), primary_key=True),
            sa.Column("name", sa.Text, nullable=False),
            sa.Column("meta", sa.JSON, nullable=False),
            sa.Column("tree", sa.JSON, nullable=False),
            sa.Column("pages", sa.JSON, nullable=False),
            sa.Column("created_at", sa.String(64), nullable=False),
        )
        self._library = sa.Table(
            "pageindex_library", metadata,
            sa.Column("key", sa.String(64), primary_key=True),
            sa.Column("value", sa.JSON, nullable=False),
        )
        self._lock_row = sa.Table(
            "pageindex_store_lock", metadata,
            sa.Column("id", sa.Integer, primary_key=True, autoincrement=False),
            sa.Column("generation", sa.Integer, nullable=False),
        )
        metadata.create_all(self._engine)
        with self._engine.begin() as conn:
            if conn.execute(sa.select(self._lock_row.c.id)).first() is None:
                try:
                    with conn.begin_nested():
                        conn.execute(self._lock_row.insert().values(
                            id=1, generation=0))
                except sa.exc.IntegrityError:  # another process got there
                    pass
        self._held = threading.local()

    @contextmanager
    def _conn(self):
        conn = getattr(self._held, "conn", None)
        if conn is not None:  # inside lock(): join its transaction
            yield conn
            return
        with self._engine.begin() as conn:
            yield conn

    @contextmanager
    def lock(self):
        if getattr(self._held, "conn", None) is not None:
            yield
            return
        with self._engine.begin() as conn:
            conn.execute(self._lock_row.update()
                         .where(self._lock_row.c.id == 1)
                         .values(generation=self._lock_row.c.generation + 1))
            self._held.conn = conn
            try:
                yield
            finally:
                self._held.conn = None

    def get_library_index(self) -> dict | None:
        with self._conn() as conn:
            index = conn.execute(sa.select(self._library.c.value).where(
                self._library.c.key == "topics")).scalar_one_or_none()
        return index if isinstance(index, dict) else None

    def save_library_index(self, index: dict) -> None:
        library = self._library
        with self._conn() as conn:
            if not conn.execute(library.update()
                                .where(library.c.key == "topics")
                                .values(value=index)).rowcount:
                conn.execute(library.insert().values(key="topics", value=index))

    # ── documents ──
    def save_document(self, doc_id: str, meta: dict, tree: list, pages: list) -> None:
        if not _is_safe_id(doc_id):
            raise ValueError(f"Invalid doc_id: {doc_id!r}")
        row = {"name": _scrub(meta.get("name") or ""), "meta": meta,
               "tree": tree, "pages": pages,
               "created_at": _scrub(meta.get("createdAt") or "")}
        docs = self._docs
        with self._conn() as conn:
            updated = conn.execute(docs.update()
                                   .where(docs.c.doc_id == doc_id)
                                   .values(**row)).rowcount
            if not updated:
                conn.execute(docs.insert().values(doc_id=doc_id, **row))

    def _read(self, doc_id: str, column):
        if not _is_safe_id(doc_id):
            return None
        with self._conn() as conn:
            return conn.execute(sa.select(column)
                                .where(self._docs.c.doc_id == doc_id)
                                ).scalar_one_or_none()

    def get_meta(self, doc_id: str) -> dict | None:
        meta = self._read(doc_id, self._docs.c.meta)
        return meta if _is_valid_meta(meta, doc_id) else None

    def get_tree(self, doc_id: str) -> list | None:
        return self._read(doc_id, self._docs.c.tree)

    def get_pages(self, doc_id: str) -> list | None:
        return self._read(doc_id, self._docs.c.pages)

    def list_metas(self) -> list[dict]:
        with self._conn() as conn:
            rows = conn.execute(sa.select(self._docs.c.doc_id,
                                          self._docs.c.meta)).all()
        return [meta for doc_id, meta in rows if _is_valid_meta(meta, doc_id)]

    def delete_document(self, doc_id: str) -> bool:
        if not _is_safe_id(doc_id):
            return False
        with self._conn() as conn:
            return conn.execute(self._docs.delete().where(
                self._docs.c.doc_id == doc_id)).rowcount > 0
