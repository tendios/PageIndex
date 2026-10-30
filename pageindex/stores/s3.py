"""S3 (or S3-compatible: MinIO, R2, …) document store."""
from __future__ import annotations

import json
import logging
import random
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from .base import DocumentStore, _is_safe_id, _is_valid_meta

logger = logging.getLogger(__name__)

_FILES = ("tree.json", "pages.json", "doc.json")  # doc.json last: it marks the document


def _dumps(data) -> bytes:
    # errors=: a lone surrogate is replaced, as the file store does.
    return json.dumps(data, ensure_ascii=False).encode("utf-8", "replace")


def _error_code(exc) -> str:
    return str(getattr(exc, "response", {}).get("Error", {}).get("Code", ""))


class S3DocumentStore(DocumentStore):
    """One object per file, the file store's layout under ``prefix``:
    ``<prefix>/docs/<doc_id>/{doc,tree,pages}.json`` (and the library
    index at ``<prefix>/library/topics.json``) — a ``.pageindex``
    directory synced to the bucket (``aws s3 sync``) reads as-is. No
    manifest: ``list_metas`` lists the prefix and reads every doc.json.

    ``client`` is any boto3 S3 client; without one it is built from the
    keyword arguments, and whatever they leave out comes from boto3's own
    chain (``AWS_ACCESS_KEY_ID``, ``AWS_ENDPOINT_URL_S3``, profiles, …).
    For MinIO pass ``endpoint_url`` and credentials.

    ``lock()`` is a lease: a ``<prefix>/.lock`` object created with
    ``If-None-Match: *``; a holder older than ``lock_ttl`` seconds is
    presumed dead and its lease taken over with ``If-Match``.
    """

    def __init__(self, bucket: str, prefix: str = "", client=None, *,
                 endpoint_url: str | None = None,
                 region_name: str | None = None,
                 aws_access_key_id: str | None = None,
                 aws_secret_access_key: str | None = None,
                 aws_session_token: str | None = None,
                 lock_ttl: float = 60.0):
        if not isinstance(bucket, str) or not bucket:
            raise ValueError("S3DocumentStore needs a bucket name.")
        if client is None:
            try:
                import boto3
            except ImportError:
                raise ImportError(
                    "S3DocumentStore needs boto3 — "
                    'pip install "pageindex[s3]".') from None
            options = {"endpoint_url": endpoint_url,
                       "region_name": region_name,
                       "aws_access_key_id": aws_access_key_id,
                       "aws_secret_access_key": aws_secret_access_key,
                       "aws_session_token": aws_session_token}
            client = boto3.client(
                "s3", **{k: v for k, v in options.items() if v is not None})
        self._s3 = client
        self._bucket = bucket
        prefix = (prefix or "").strip("/")
        self._prefix = prefix + "/" if prefix else ""
        self._lock_ttl = lock_ttl

    def _key(self, *parts: str) -> str:
        return self._prefix + "/".join(parts)

    def _put(self, key: str, data, **conditions):
        return self._s3.put_object(Bucket=self._bucket, Key=key,
                                   Body=_dumps(data),
                                   ContentType="application/json",
                                   **conditions)

    def _get(self, key: str):
        try:
            body = self._s3.get_object(Bucket=self._bucket, Key=key)["Body"]
            return json.loads(body.read())
        except ValueError:
            logger.warning("Unreadable JSON at s3://%s/%s; treating it as absent",
                           self._bucket, key)
            return None
        except Exception as e:
            if _error_code(e) in ("NoSuchKey", "404", "NotFound"):
                return None
            raise

    def _exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self._bucket, Key=key)
            return True
        except Exception as e:
            if _error_code(e) in ("NoSuchKey", "404", "NotFound"):
                return False
            raise

    def _list(self, prefix: str) -> list[str]:
        keys = []
        paginator = self._s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket, Prefix=prefix):
            keys.extend(item["Key"] for item in page.get("Contents", ()))
        return keys

    # ── lock ──
    def _try_acquire(self, token: str) -> bool:
        key = self._key(".lock")
        try:
            self._put(key, {"token": token}, IfNoneMatch="*")
            return True
        except Exception as e:
            if _error_code(e) not in ("PreconditionFailed",
                                      "ConditionalRequestConflict"):
                raise
        try:
            held = self._s3.get_object(Bucket=self._bucket, Key=key)
        except Exception as e:
            if _error_code(e) in ("NoSuchKey", "404", "NotFound"):
                return False  # released meanwhile: retry the create
            raise
        # Age by the server's clock when it says (Date), else ours.
        date = held["ResponseMetadata"].get("HTTPHeaders", {}).get("date")
        now = parsedate_to_datetime(date) if date else datetime.now(timezone.utc)
        if (now - held["LastModified"]).total_seconds() < self._lock_ttl:
            return False
        logger.warning("Taking over a stale lock at s3://%s/%s", self._bucket, key)
        try:
            self._put(key, {"token": token}, IfMatch=held["ETag"])
            return True
        except Exception as e:
            if _error_code(e) in ("PreconditionFailed", "NoSuchKey",
                                  "ConditionalRequestConflict", "404"):
                return False
            raise

    @contextmanager
    def lock(self):
        token = uuid.uuid4().hex
        delay = 0.05
        while not self._try_acquire(token):
            time.sleep(delay * (1 + random.random()))
            delay = min(delay * 2, 1.0)
        try:
            yield
        finally:
            key = self._key(".lock")
            try:
                held = self._s3.get_object(Bucket=self._bucket, Key=key)
                if json.loads(held["Body"].read()).get("token") == token:
                    self._s3.delete_object(Bucket=self._bucket, Key=key)
            except Exception as e:  # expired and taken over: not ours to drop
                logger.warning("Could not release s3://%s/%s: %s",
                               self._bucket, key, e)

    def get_library_index(self) -> dict | None:
        index = self._get(self._key("library", "topics.json"))
        return index if isinstance(index, dict) else None

    def save_library_index(self, index: dict) -> None:
        self._put(self._key("library", "topics.json"), index)

    # ── documents ──
    def save_document(self, doc_id: str, meta: dict, tree: list, pages: list) -> None:
        if not _is_safe_id(doc_id):
            raise ValueError(f"Invalid doc_id: {doc_id!r}")
        for name, data in zip(_FILES, (tree, pages, meta)):
            self._put(self._key("docs", doc_id, name), data)

    def _read_doc_file(self, doc_id: str, name: str):
        if not _is_safe_id(doc_id) or not self._exists(
                self._key("docs", doc_id, "doc.json")):
            return None
        return self._get(self._key("docs", doc_id, name))

    def get_meta(self, doc_id: str) -> dict | None:
        if not _is_safe_id(doc_id):
            return None
        meta = self._get(self._key("docs", doc_id, "doc.json"))
        return meta if _is_valid_meta(meta, doc_id) else None

    def get_tree(self, doc_id: str) -> list | None:
        return self._read_doc_file(doc_id, "tree.json")

    def get_pages(self, doc_id: str) -> list | None:
        return self._read_doc_file(doc_id, "pages.json")

    def list_metas(self) -> list[dict]:
        root = self._key("docs", "")
        ids = []
        for key in self._list(root):
            doc_id, _, name = key[len(root):].partition("/")
            if name == "doc.json" and _is_safe_id(doc_id):
                ids.append(doc_id)
        if not ids:
            return []
        with ThreadPoolExecutor(max_workers=min(16, len(ids))) as pool:
            metas = list(pool.map(self.get_meta, ids))
        return [meta for meta in metas if meta is not None]

    def delete_document(self, doc_id: str) -> bool:
        if not _is_safe_id(doc_id):
            return False
        keys = self._list(self._key("docs", doc_id, ""))
        marker = self._key("docs", doc_id, "doc.json")
        existed = marker in keys
        if existed:  # first, so a half-deleted document is never listed
            self._s3.delete_object(Bucket=self._bucket, Key=marker)
        rest = [key for key in keys if key != marker]
        for i in range(0, len(rest), 1000):
            self._s3.delete_objects(Bucket=self._bucket, Delete={
                "Objects": [{"Key": key} for key in rest[i:i + 1000]],
                "Quiet": True})
        return existed
