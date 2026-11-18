"""submit_pages: a document from pages the caller already has, stored as an indexed one, with no file to read and
no model call; every read of the client sees it as any other document."""
import json

import pytest

import pageindex.utils
from pageindex import PageIndexAPIError, PageIndexClient

SHEETS = [
    "## Presupuesto\n\n| Partida | Importe |\n| --- | --- |\n| Excavación | 1200 |",
    "## Presupuesto (filas 3-4)\n\n| Partida | Importe |\n| --- | --- |\n| Hormigón | 800 |",
    "## Resumen\n\n| Total |\n| --- |\n| 2000 |",
]
TREE = [
    {"title": "Presupuesto", "start_index": 1, "end_index": 2},
    {"title": "Resumen", "start_index": 3, "end_index": 3, "summary": "El total."},
]


@pytest.fixture
def client(tmp_path):
    return PageIndexClient(storage_path=str(tmp_path / "store"))


@pytest.fixture(autouse=True)
def no_model(monkeypatch):
    """Any model call fails the test: submit_pages calls none."""

    def refuse(*args, **kwargs):
        raise AssertionError("submit_pages called a model")

    monkeypatch.setattr(pageindex.utils, "llm_completion", refuse)
    monkeypatch.setattr(pageindex.utils, "llm_acompletion", refuse)


def test_the_pages_and_the_tree_are_stored_as_given_and_read_back(client):
    tree = json.loads(json.dumps(TREE))
    result = client.submit_pages("presupuesto.xlsx", SHEETS, tree, description="Dos hojas.",
                                 metadata={"tenderId": "t1"}, doc_id="sheet-1")
    assert result["doc_id"] == "sheet-1" and result["name"] == "presupuesto.xlsx"
    assert result["usage"]["total_tokens"] == 0 and set(result["timings"]) == {"save", "total"}
    assert tree == TREE  # the caller's nodes are not touched
    assert client.get_document("sheet-1") == {
        "id": "sheet-1", "name": "presupuesto.xlsx", "description": "Dos hojas.", "status": "completed",
        "createdAt": client.get_document("sheet-1")["createdAt"], "pageNum": 3, "folderId": None,
        "metadata": {"tenderId": "t1"}}
    assert client.get_tree("sheet-1", node_summary=True, include_text=False)["result"] == [
        {"title": "Presupuesto", "node_id": "0000", "start_index": 1, "end_index": 2},
        {"title": "Resumen", "node_id": "0001", "start_index": 3, "end_index": 3, "summary": "El total."},
    ]
    with_text = client.get_tree("sheet-1")["result"]
    assert "Excavación" in with_text[0]["text"] and "Hormigón" in with_text[0]["text"]
    assert [p["markdown"] for p in client.get_page_content("sheet-1", "1-3")] == SHEETS
    assert client.get_ocr("sheet-1")["result"] == [
        {"page_index": i + 1, "markdown": text} for i, text in enumerate(SHEETS)]
    assert client._api._store.get_meta("sheet-1")["mode"] == "pages"


def test_nested_nodes_and_one_node_per_page_without_a_tree(client):
    nested = [{"title": "Libro", "start_index": 1, "end_index": 3,
               "nodes": [{"title": "Hoja 1", "start_index": 1, "end_index": 2},
                         {"title": "Hoja 2", "start_index": 3, "end_index": 3}]}]
    doc = client.submit_pages("libro.xlsx", SHEETS, nested)["doc_id"]
    tree = client.get_tree(doc, include_text=False)["result"]
    assert [n["node_id"] for n in [tree[0], *tree[0]["nodes"]]] == ["0000", "0001", "0002"]

    plain = client.submit_pages("plano.txt", ["uno", "dos"])["doc_id"]
    assert client.get_tree(plain, include_text=False)["result"] == [
        {"title": "Page 1", "node_id": "0000", "start_index": 1, "end_index": 1},
        {"title": "Page 2", "node_id": "0001", "start_index": 2, "end_index": 2},
    ]


def test_a_taken_id_is_refused_and_replace_replaces_it(client):
    client.submit_pages("a.xlsx", SHEETS, TREE, doc_id="d")
    with pytest.raises(PageIndexAPIError) as taken:
        client.submit_pages("a.xlsx", SHEETS, TREE, doc_id="d")
    assert taken.value.status_code == 409
    client.submit_pages("a.xlsx", ["## Otra\n\nnada"], [{"title": "Otra", "start_index": 1, "end_index": 1}],
                        doc_id="d", if_exists="replace", metadata={"v": 2})
    assert client.get_document("d")["pageNum"] == 1 and client.get_document("d")["metadata"] == {"v": 2}
    assert client.get_document("d")["name"] == "a.xlsx"  # the document replaced does not hold its own name
    assert client.list_documents()["total"] == 1


def test_a_taken_name_gains_a_suffix(client):
    client.submit_pages("a.xlsx", SHEETS, TREE)
    with pytest.warns(UserWarning, match='stored as "a_1.xlsx"'):
        assert client.submit_pages("a.xlsx", SHEETS, TREE)["name"] == "a_1.xlsx"


@pytest.mark.parametrize(("arguments", "message"), [
    ({"name": " "}, "name must be a non-empty string"),
    ({"pages": []}, "pages must be a non-empty list of strings"),
    ({"pages": ["uno", 2]}, "pages must be a non-empty list of strings"),
    ({"structure": []}, "structure must be a non-empty list"),
    ({"structure": [{"title": "x", "start_index": 0, "end_index": 1}]}, "1 <= start_index"),
    ({"structure": [{"title": "x", "start_index": 2, "end_index": 1}]}, "1 <= start_index"),
    ({"structure": [{"title": "x", "start_index": 1, "end_index": 4}]}, "<= 3"),
    ({"structure": [{"title": "x", "start_index": True, "end_index": 1}]}, "integers"),
    ({"structure": [{"title": "", "start_index": 1, "end_index": 1}]}, "needs a title"),
    ({"structure": [{"title": "x", "start_index": 1, "end_index": 1, "text": "y"}]}, "unknown keys \\['text'\\]"),
    ({"structure": [{"title": "x", "start_index": 1, "end_index": 3,
                     "nodes": [{"title": "y", "start_index": 1}]}]}, "structure\\[0\\].nodes\\[0\\]"),
    ({"structure": [{"title": "x", "start_index": 1, "end_index": 1, "summary": 3}]}, "summary"),
    ({"description": 3}, "description must be a string"),
    ({"metadata": ["no"]}, "metadata must be a dict"),
    ({"doc_id": "a/b"}, "invalid doc_id"),
    ({"if_exists": "replace"}, "if_exists applies to a given doc_id only"),
])
def test_invalid_arguments_are_refused_before_anything_is_stored(client, arguments, message):
    call = {"name": "a.xlsx", "pages": SHEETS, "structure": TREE, **arguments}
    with pytest.raises(PageIndexAPIError, match=message):
        client.submit_pages(call.pop("name"), call.pop("pages"), call.pop("structure"), **call)
    assert client.list_documents()["total"] == 0


def test_blank_pages_are_refused_as_a_blank_pdf_is(client):
    with pytest.raises(PageIndexAPIError, match="no content") as blank:
        client.submit_pages("vacia.xlsx", [" ", "\n"])
    assert blank.value.status_code is None


def test_the_cloud_client_refuses_it():
    cloud = PageIndexClient(api_key="pi-test")
    with pytest.raises(PageIndexAPIError, match="local-only"):
        cloud.submit_pages("a.xlsx", SHEETS, TREE)


def test_any_store_keeps_it(tmp_path):
    pytest.importorskip("sqlalchemy")
    client = PageIndexClient(index={"store": {"type": "sql", "url": f"sqlite:///{tmp_path / 'docs.db'}"}})
    doc = client.submit_pages("a.xlsx", SHEETS, TREE)["doc_id"]
    assert [p["markdown"] for p in client.get_page_content(doc, "2")] == [SHEETS[1]]
