"""Document stores: one contract, three backends (disk, S3 via moto,
SQLite via SQLAlchemy), plus copy() between them and the client's
index={"store": ...} wiring. No network."""
import json
import threading
import warnings
import time
from pathlib import Path

import pytest

from pageindex import PageIndexAPIError, PageIndexClient
from pageindex.local_api import LocalAPI
from pageindex.stores import (DocumentStore, FileDocumentStore, copy,
                              open_store)

BUCKET = "pageindex-test"


def make_meta(doc_id, name="report.pdf", created_at="2026-08-01T10:00:00.000"):
    return {"id": doc_id, "name": name, "description": "d",
            "status": "completed", "createdAt": created_at, "pageNum": 2,
            "folderId": None, "metadata": {"k": "v"}, "mode": "flash"}


TREE = [{"title": "Intro", "node_id": "0000", "start_index": 1,
         "end_index": 2, "summary": "s — ünïcode",
         "nodes": [{"title": "Child", "node_id": "0001",
                    "start_index": 2, "end_index": 2}]}]
PAGES = [{"page_index": 1, "markdown": "# one"},
         {"page_index": 2, "markdown": "two ✓"}]


@pytest.fixture
def s3_client():
    pytest.importorskip("moto")
    import boto3
    from moto import mock_aws
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1",
                              aws_access_key_id="test",
                              aws_secret_access_key="test")
        client.create_bucket(Bucket=BUCKET)
        yield client


def _file_store(request, tmp_path):
    return FileDocumentStore(str(tmp_path / "store"))


def _s3_store(request, tmp_path):
    from pageindex.stores import S3DocumentStore
    return S3DocumentStore(BUCKET, prefix="idx/",
                           client=request.getfixturevalue("s3_client"))


def _sql_store(request, tmp_path):
    pytest.importorskip("sqlalchemy")
    from pageindex.stores import SqlDocumentStore
    return SqlDocumentStore("sqlite://")


FACTORIES = {"file": _file_store, "s3": _s3_store, "sql": _sql_store}


@pytest.fixture(params=sorted(FACTORIES))
def store(request, tmp_path):
    return FACTORIES[request.param](request, tmp_path)


# ── the shared contract ──

def test_save_and_read_back(store):
    store.save_document("pi-a", make_meta("pi-a"), TREE, PAGES)
    assert store.get_meta("pi-a") == make_meta("pi-a")
    assert store.get_tree("pi-a") == TREE
    assert store.get_pages("pi-a") == PAGES


def test_missing_document(store):
    assert store.get_meta("pi-none") is None
    assert store.get_tree("pi-none") is None
    assert store.get_pages("pi-none") is None
    assert store.delete_document("pi-none") is False
    assert store.list_metas() == []


def test_list_and_delete(store):
    store.save_document("pi-a", make_meta("pi-a", "a.pdf"), TREE, PAGES)
    store.save_document("pi-b", make_meta("pi-b", "b.pdf"), TREE, PAGES)
    assert sorted(m["id"] for m in store.list_metas()) == ["pi-a", "pi-b"]
    assert store.delete_document("pi-a") is True
    assert store.delete_document("pi-a") is False
    assert [m["id"] for m in store.list_metas()] == ["pi-b"]
    assert store.get_meta("pi-a") is None
    assert store.get_tree("pi-a") is None
    assert store.get_pages("pi-a") is None


def test_unsafe_ids(store):
    with pytest.raises(ValueError, match="Invalid doc_id"):
        store.save_document("../x", make_meta("../x"), TREE, PAGES)
    for bad in ("", "..", "a/b", "a\\b"):
        assert store.get_meta(bad) is None
        assert store.delete_document(bad) is False


def test_invalid_meta_is_not_listed(store):
    store.save_document("pi-a", {"id": "pi-a"}, TREE, PAGES)
    assert store.get_meta("pi-a") is None
    assert store.list_metas() == []


def test_lone_surrogate_is_replaced(store):
    meta = make_meta("pi-a", name="bad-\udcff.pdf")
    store.save_document("pi-a", meta, TREE, PAGES)
    assert store.get_meta("pi-a")["name"] == "bad-?.pdf"


def test_duplicate_name_gets_a_suffix(store):
    api = LocalAPI(None, model="m", summary_model="m", store=store)
    store.save_document("pi-a", make_meta("pi-a", "report.pdf"), TREE, PAGES)
    with store.lock():
        assert api._unique_doc_name("report.pdf") == "report_1.pdf"
        store.save_document("pi-b", make_meta("pi-b", "report_1.pdf"),
                            TREE, PAGES)
        assert api._unique_doc_name("report.pdf") == "report_2.pdf"
        assert api._unique_doc_name("other.pdf") == "other.pdf"


def test_client_lists_through_the_store(store):
    for i in range(3):
        store.save_document(f"pi-{i}", make_meta(
            f"pi-{i}", f"{i}.pdf", f"2026-08-0{i + 1}T10:00:00.000"),
            TREE, PAGES)
    client = PageIndexClient(index={"store": store})
    assert client.storage_path is None
    page = client.list_documents(limit=2, offset=0)
    assert page["total"] == 3
    assert [d["id"] for d in page["documents"]] == ["pi-2", "pi-1"]
    assert client.get_document("pi-0")["name"] == "0.pdf"
    client.delete_document("pi-0")
    assert store.get_meta("pi-0") is None


def test_library_index_round_trip(store):
    assert store.get_library_index() is None
    index = {"version": 1, "topics": {"t ✓": [{"doc_id": "pi-a",
                                               "node_id": "0001"}]}}
    store.save_library_index(index)
    assert store.get_library_index() == index
    store.save_library_index({"version": 1, "topics": {}})
    assert store.get_library_index() == {"version": 1, "topics": {}}
    assert store.list_metas() == []  # not a document


# ── lock() ──

def test_lock_excludes_other_holders(request, tmp_path):
    factories = dict(FACTORIES)

    def sql_file(request, tmp_path):  # threads need a real file, not :memory:
        pytest.importorskip("sqlalchemy")
        from pageindex.stores import SqlDocumentStore
        return SqlDocumentStore(f"sqlite:///{tmp_path / 'store.db'}")

    factories["sql"] = sql_file
    for kind, factory in sorted(factories.items()):
        (tmp_path / kind).mkdir()
        store = factory(request, tmp_path / kind)
        order = []

        def contender():
            with store.lock():
                order.append("second")

        with store.lock():
            thread = threading.Thread(target=contender)
            thread.start()
            time.sleep(0.3)
            order.append("first")
        thread.join(timeout=10)
        assert order == ["first", "second"], kind


def test_s3_lock_takes_over_a_stale_lease(s3_client):
    from pageindex.stores import S3DocumentStore
    s3_client.put_object(Bucket=BUCKET, Key="idx/.lock",
                         Body=b'{"token": "dead"}')
    store = S3DocumentStore(BUCKET, "idx", client=s3_client, lock_ttl=0)
    with store.lock():
        held = json.loads(s3_client.get_object(
            Bucket=BUCKET, Key="idx/.lock")["Body"].read())
        assert held["token"] != "dead"
    assert "Contents" not in s3_client.list_objects_v2(
        Bucket=BUCKET, Prefix="idx/.lock")


def test_s3_release_leaves_a_lease_taken_over_by_another(s3_client):
    from pageindex.stores import S3DocumentStore
    store = S3DocumentStore(BUCKET, "idx", client=s3_client)
    with store.lock():
        s3_client.put_object(Bucket=BUCKET, Key="idx/.lock",
                             Body=b'{"token": "other"}')
    held = s3_client.get_object(Bucket=BUCKET, Key="idx/.lock")
    assert json.loads(held["Body"].read()) == {"token": "other"}


# ── one format everywhere ──

def test_s3_reads_a_synced_disk_store(tmp_path, s3_client):
    """`aws s3 sync .pageindex s3://bucket/prefix` is a migration."""
    from pageindex.stores import S3DocumentStore
    disk = FileDocumentStore(str(tmp_path))
    disk.save_document("pi-a", make_meta("pi-a"), TREE, PAGES)
    disk.list_metas()  # writes manifest.json too: S3 ignores it
    for path in tmp_path.rglob("*.json"):
        s3_client.put_object(Bucket=BUCKET,
                             Key="idx/" + path.relative_to(tmp_path).as_posix(),
                             Body=path.read_bytes())
    s3 = S3DocumentStore(BUCKET, "idx", client=s3_client)
    assert s3.list_metas() == [make_meta("pi-a")]
    assert s3.get_tree("pi-a") == TREE
    stored = s3_client.get_object(Bucket=BUCKET, Key="idx/docs/pi-a/doc.json")
    assert json.loads(stored["Body"].read()) == json.loads(
        (tmp_path / "docs" / "pi-a" / "doc.json").read_text())


def test_copy_round_trip(request, tmp_path):
    stores = [FACTORIES[kind](request, tmp_path / kind)
              for kind in ("file", "s3", "sql")]
    stores.append(FileDocumentStore(str(tmp_path / "back")))
    stores[0].save_document("pi-a", make_meta("pi-a", "a.pdf"), TREE, PAGES)
    stores[0].save_document("pi-b", make_meta("pi-b", "b.pdf"), TREE, PAGES)
    for source, target in zip(stores, stores[1:]):
        assert sorted(copy(source, target)) == ["pi-a", "pi-b"]
    back = stores[-1]
    assert back.get_meta("pi-a") == make_meta("pi-a", "a.pdf")
    assert back.get_pages("pi-b") == PAGES
    for name in ("doc.json", "tree.json", "pages.json"):
        assert (Path(tmp_path / "back/docs/pi-a" / name).read_bytes()
                == Path(tmp_path / "file/store/docs/pi-a" / name).read_bytes())
    assert back.get_library_index() is None
    stores[0].save_library_index({"version": 1, "topics": {"t": []}})
    copy(stores[0], stores[1])
    assert stores[1].get_library_index() == {"version": 1, "topics": {"t": []}}
    assert copy(stores[0], back) == []  # already there
    assert sorted(copy(stores[0], back, overwrite=True)) == ["pi-a", "pi-b"]


def test_copy_takes_dict_specs(tmp_path):
    FileDocumentStore(str(tmp_path / "a")).save_document(
        "pi-a", make_meta("pi-a"), TREE, PAGES)
    assert copy({"type": "file", "path": str(tmp_path / "a")},
                {"type": "file", "path": str(tmp_path / "b")}) == ["pi-a"]
    assert FileDocumentStore(str(tmp_path / "b")).get_tree("pi-a") == TREE


# ── configuration ──

def test_default_stays_on_disk(tmp_path):
    client = PageIndexClient(storage_path=str(tmp_path / "s"))
    assert isinstance(client._api._store, FileDocumentStore)
    assert client.storage_path == str(tmp_path / "s")


def test_store_from_dict(tmp_path):
    client = PageIndexClient(index={"store": {"type": "file",
                                              "path": str(tmp_path / "s")}})
    assert isinstance(client._api._store, FileDocumentStore)


def test_store_from_sql_dict():
    pytest.importorskip("sqlalchemy")
    from pageindex.stores import SqlDocumentStore
    client = PageIndexClient(index={"mode": "local",
                                    "store": {"type": "sql", "url": "sqlite://"}})
    assert isinstance(client._api._store, SqlDocumentStore)


def test_store_from_s3_dict(s3_client):
    from pageindex.stores import S3DocumentStore
    client = PageIndexClient(index={"store": {
        "type": "s3", "bucket": BUCKET, "region_name": "us-east-1"}})
    assert isinstance(client._api._store, S3DocumentStore)
    assert client.list_documents()["total"] == 0


@pytest.mark.parametrize("index, match", [
    ({"store": {"type": "file", "path": "x"}, "storage_path": "y"},
     r'index\["store"\] and storage_path'),
    ({"store": {"type": "ftp"}}, '"type" must be "file", "s3" or "sql"'),
    ({"store": {"type": "file"}}, "storage_dir"),
    ({"store": {"type": "sql", "url": "sqlite://", "nope": 1}}, "nope"),
    ({"store": "s3://bucket"}, r'index\["store"\] must be a DocumentStore'),
    ({"store": {}}, "empty"),
    ({"api_key": "k", "store": {"type": "file", "path": "x"}}, "mixes cloud"),
])
def test_store_config_errors(index, match):
    with pytest.raises(PageIndexAPIError, match=match):
        PageIndexClient(index=index)


def test_sql_without_sqlalchemy_says_how_to_install(monkeypatch):
    pytest.importorskip("sqlalchemy")
    import pageindex.stores.sql as sql_module
    monkeypatch.setattr(sql_module, "sa", None)
    with pytest.raises(PageIndexAPIError, match=r'pageindex\[sql\]'):
        PageIndexClient(index={"store": {"type": "sql", "url": "sqlite://"}})


def test_custom_store_subclass(tmp_path):
    class Wrapped(FileDocumentStore):
        pass

    store = Wrapped(str(tmp_path))
    assert isinstance(store, DocumentStore)
    assert PageIndexClient(index={"store": store})._api._store is store
    assert open_store(store) is store


def test_old_import_path_still_works():
    from pageindex.local_store import DocStore
    assert DocStore is FileDocumentStore


# ── caller-chosen doc ids (local submit_document(doc_id=...)) ──

STRUCTURE = [{"title": "Root", "node_id": "0000", "start_index": 1,
              "end_index": 2, "summary": "root summary", "text": "root text"}]


@pytest.fixture
def indexing(monkeypatch):
    """A stubbed standard pipeline; set indexing.title to change the tree."""
    import importlib
    classic = importlib.import_module("pageindex.page_index_classic")
    state = {"title": "Root", "calls": 0}

    def fake_page_index_main(doc, opt=None, logger=None, page_list=None):
        state["calls"] += 1
        structure = json.loads(json.dumps(STRUCTURE))
        structure[0]["title"] = state["title"]
        return {"doc_name": "sample.pdf", "doc_description": "d",
                "structure": structure}
    monkeypatch.setattr(classic, "page_index_main", fake_page_index_main)
    return state


def test_external_doc_id_is_used_everywhere(store, sample_pdf, indexing):
    client = PageIndexClient(index={"store": store})
    result = client.submit_document(sample_pdf, mode="standard",
                                    doc_id="contract-2026_v2.1")
    assert result["doc_id"] == "contract-2026_v2.1"
    assert store.get_meta("contract-2026_v2.1")["id"] == "contract-2026_v2.1"
    assert client.get_document("contract-2026_v2.1")["name"] == "sample.pdf"
    assert client.get_tree("contract-2026_v2.1")["result"][0]["title"] == "Root"
    assert "Hello page one" in client.get_page_content(
        "contract-2026_v2.1", "1")[0]["markdown"]
    assert client.get_citations('<cite doc="sample.pdf" page="1"/>') == [
        {"document": "sample.pdf", "doc_id": "contract-2026_v2.1", "page": 1}]
    client.delete_document("contract-2026_v2.1")
    assert store.list_metas() == []


@pytest.mark.parametrize("doc_id", [
    "", "a" * 129, "a/b", "a\\b", "a b", "..", ".", "ñ", "x\n", 7])
def test_invalid_external_doc_id(tmp_path, sample_pdf, indexing, doc_id):
    client = PageIndexClient(storage_path=str(tmp_path / "s"))
    with pytest.raises(PageIndexAPIError, match="invalid doc_id") as info:
        client.submit_document(sample_pdf, mode="standard", doc_id=doc_id)
    assert info.value.status_code == 400
    assert indexing["calls"] == 0


def test_taken_external_doc_id_is_refused_before_indexing(store, sample_pdf,
                                                          indexing):
    client = PageIndexClient(index={"store": store})
    client.submit_document(sample_pdf, mode="standard", doc_id="doc-1")
    with pytest.raises(PageIndexAPIError, match="already exists") as info:
        client.submit_document(sample_pdf, mode="standard", doc_id="doc-1")
    assert info.value.status_code == 409
    assert indexing["calls"] == 1
    with pytest.raises(PageIndexAPIError, match="already exists"):
        client.submit_document(sample_pdf, mode="standard", doc_id="doc-1",
                               if_exists="error")


def test_if_exists_replace_leaves_nothing_of_the_old_document(
        store, sample_pdf, indexing):
    client = PageIndexClient(index={"store": store})
    client.submit_document(sample_pdf, mode="standard", doc_id="doc-1",
                           metadata={"version": 1})
    indexing["title"] = "Replaced"
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # its own name is free: no rename
        result = client.submit_document(sample_pdf, mode="standard",
                                        doc_id="doc-1", if_exists="replace")
    assert result == {**result, "doc_id": "doc-1", "name": "sample.pdf"}
    assert [m["id"] for m in store.list_metas()] == ["doc-1"]
    assert store.get_meta("doc-1")["metadata"] is None
    assert client.get_tree("doc-1")["result"][0]["title"] == "Replaced"
    assert [p["page_index"] for p in store.get_pages("doc-1")] == [1, 2]
    # replace also works for an id that is not there yet
    client.submit_document(sample_pdf, mode="standard", doc_id="doc-2",
                           if_exists="replace")
    assert sorted(m["id"] for m in store.list_metas()) == ["doc-1", "doc-2"]


def test_external_ids_keep_name_uniquing(tmp_path, sample_pdf, indexing):
    client = PageIndexClient(storage_path=str(tmp_path / "s"))
    client.submit_document(sample_pdf, mode="standard", doc_id="a")
    with pytest.warns(UserWarning, match="sample_1.pdf"):
        assert client.submit_document(sample_pdf, mode="standard",
                                      doc_id="b")["name"] == "sample_1.pdf"


@pytest.mark.parametrize("kwargs, match", [
    ({"if_exists": "replace"}, "doc_id only"),
    ({"doc_id": "a", "if_exists": "merge"}, "if_exists must be"),
])
def test_if_exists_needs_a_doc_id_and_a_known_value(tmp_path, sample_pdf,
                                                    indexing, kwargs, match):
    client = PageIndexClient(storage_path=str(tmp_path / "s"))
    with pytest.raises(PageIndexAPIError, match=match):
        client.submit_document(sample_pdf, mode="standard", **kwargs)


def test_cloud_refuses_external_doc_id(sample_pdf):
    client = PageIndexClient(api_key="k")
    with pytest.raises(PageIndexAPIError, match="doc_id is local-only"):
        client.submit_document(sample_pdf, doc_id="a")
