"""Keep local-mode indexes in S3 (or MinIO) or Postgres instead of on disk.

    pip install -U "pageindex[s3,sql]" psycopg

S3 credentials come from boto3's usual chain (AWS_ACCESS_KEY_ID /
AWS_SECRET_ACCESS_KEY, AWS_ENDPOINT_URL_S3 for MinIO, or a profile). Set
the bucket and the database URL below, then:

    python pageindex-storage-backends.py report.pdf
"""
import os
import sys

from pageindex import PageIndexClient
from pageindex.stores import S3DocumentStore, SqlDocumentStore, copy

BUCKET = os.environ.get("PAGEINDEX_BUCKET", "my-pageindex-bucket")
DATABASE_URL = os.environ.get(
    "PAGEINDEX_DATABASE_URL", "postgresql+psycopg://pageindex:pageindex@localhost/pageindex")

# 1. Index straight into S3: <prefix>/docs/<doc_id>/{doc,tree,pages}.json
s3_client = PageIndexClient(index={"model": "gpt-6-luna", "store": {
    "type": "s3", "bucket": BUCKET, "prefix": "pageindex",
    # MinIO: "endpoint_url": "http://localhost:9000",
}}, chat="gpt-6-sol")
doc_id = s3_client.submit_document(sys.argv[1])["doc_id"]
print(s3_client.chat("Summarize the document in three bullets.", doc_id=doc_id))

# 2. The same documents in Postgres: one row per document in pageindex_documents.
s3 = S3DocumentStore(BUCKET, prefix="pageindex")
pg = SqlDocumentStore(DATABASE_URL)
print("copied to Postgres:", copy(s3, pg))

pg_client = PageIndexClient(index={"model": "gpt-6-luna", "store": pg},
                            chat="gpt-6-sol")
print([d["name"] for d in pg_client.list_documents()["documents"]])

# 3. An existing ./.pageindex directory moves the same way.
print("copied from disk:", copy({"type": "file", "path": ".pageindex"}, pg))
