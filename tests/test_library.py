"""Library view for multi-document questions (local mode): the doc_id-list
targeting block, get_library_structure(), the agent guidance, and the
optional topic index. No network: chat runs on a scripted model."""
import json

import pytest

import pageindex.agent_tools as agent_tools
import pageindex.library as library
from pageindex import PageIndexAPIError, PageIndexClient
from pageindex.agent_tools import call_tool, tool_names
from pageindex.stores import FileDocumentStore

LEAF = {"title": "Leaf", "node_id": "0003", "start_index": 4,
        "end_index": 4, "summary": "leaf summary", "text": "LEAF TEXT"}


def make_tree(prefix):
    return [{"title": f"{prefix} Part I", "node_id": "0000", "start_index": 1,
             "end_index": 4, "summary": f"{prefix} part one",
             "text": "ROOT TEXT",
             "nodes": [{"title": f"{prefix} Item 1", "node_id": "0001",
                        "start_index": 2, "end_index": 4,
                        "summary": f"{prefix} item one",
                        "nodes": [dict(LEAF)]}]},
            {"title": f"{prefix} Part II", "node_id": "0002",
             "start_index": 5, "end_index": 6, "summary": f"{prefix} part two"}]


def seed(store, doc_id, name, created_at="2026-08-01T10:00:00.000"):
    meta = {"id": doc_id, "name": name, "description": f"{name} description",
            "status": "completed", "createdAt": created_at, "pageNum": 6,
            "folderId": None, "metadata": {"year": doc_id[-4:]},
            "mode": "flash"}
    pages = [{"page_index": i, "markdown": f"{name} page {i}"}
             for i in range(1, 7)]
    store.save_document(doc_id, meta, make_tree(name), pages)
    return doc_id


@pytest.fixture
def store(tmp_path):
    return FileDocumentStore(str(tmp_path / "store"))


@pytest.fixture
def client(store):
    seed(store, "pi-2022", "FY2022.pdf")
    seed(store, "pi-2025", "FY2025.pdf", "2026-08-02T10:00:00.000")
    seed(store, "pi-other", "other.pdf", "2026-08-03T10:00:00.000")
    return PageIndexClient(index={"store": store})


def view_of(block):
    """The JSON view line of a library targeting block."""
    lines = block.splitlines()
    return json.loads(lines[lines.index(next(
        line for line in lines if line.startswith("Document library"))) + 1])


# ── the targeting block ──

def test_list_targeting_carries_every_outline(client):
    block = client.document_context(["pi-2022", "pi-2025"])
    assert block.startswith(
        "The user has specified documents: FY2022.pdf, FY2025.pdf")
    view = view_of(block)
    assert [root["doc_name"] for root in view] == ["FY2022.pdf", "FY2025.pdf"]
    root = view[0]
    assert root["doc_id"] == "pi-2022"
    assert root["metadata"] == {"year": "2022"}
    assert root["description"] == "FY2022.pdf description"
    assert root["page_count"] == 6
    leaf = root["nodes"][0]["nodes"][0]["nodes"][0]
    assert leaf == {"title": "Leaf", "node_id": "0003", "start_index": 4,
                    "end_index": 4, "summary": "leaf summary"}
    assert "TEXT" not in block  # never node text
    assert "These outlines are complete" in block


def test_view_abridges_to_fit_the_budget(client):
    from pageindex.utils import count_tokens
    full = view_of(client.document_context(["pi-2022", "pi-2025"]))

    def size(levels):
        view = full if levels is None else library._abridged(full, levels)
        return count_tokens(json.dumps(view, ensure_ascii=False))

    assert size(None) > size(2) > size(1) > size(0)
    client.library_view_max_tokens = size(None) - 1
    block = client.document_context(["pi-2022", "pi-2025"])
    item = view_of(block)[0]["nodes"][0]["nodes"][0]
    assert "nodes" not in item and item["nodes_omitted"] == 1
    assert "abridged to levels 1-2 to fit" in block
    assert "get_library_structure()" in block

    client.library_view_max_tokens = size(1)
    block = client.document_context(["pi-2022", "pi-2025"])
    part = view_of(block)[0]["nodes"][0]
    assert "nodes" not in part and part["nodes_omitted"] == 2
    assert "abridged to level 1" in block

    client.library_view_max_tokens = 1  # nothing fits: names and metadata
    root = view_of(client.document_context(["pi-2022"]))[0]
    assert "nodes" not in root and root["nodes_omitted"] == 4
    assert root["doc_name"] == "FY2022.pdf"


def test_single_id_and_view_off_keep_the_old_block(client):
    single = client.document_context("pi-2022")
    assert single.startswith("The user has specified document: FY2022.pdf")
    assert "Document library" not in single
    client.library_view = False
    block = client.document_context(["pi-2022", "pi-2025"])
    assert "Documents metadata: [" in block
    assert "Document library" not in block


def test_prune_counts_what_it_hides():
    tree = make_tree("x")
    assert library.prune(tree, 3) == tree
    assert library.prune(tree, 2)[0]["nodes"][0]["nodes_omitted"] == 1
    assert library.prune(tree, 1)[0]["nodes_omitted"] == 2
    assert "nodes_omitted" not in library.prune(tree, 1)[1]


# ── get_library_structure ──

def run(client, name, doc_ids=None, **arguments):
    text, is_error = call_tool(client, name, arguments, doc_ids=doc_ids)
    return json.loads(text), is_error


def test_tool_returns_named_outlines_in_one_call(client):
    payload, is_error = run(client, "get_library_structure",
                            doc_names=["FY2025.pdf", "FY2022.pdf"])
    assert not is_error
    docs = payload["documents"]
    assert [d["doc_name"] for d in docs] == ["FY2025.pdf", "FY2022.pdf"]
    assert docs[0]["nodes"][0]["nodes"][0]["nodes"][0]["title"] == "Leaf"
    assert "TEXT" not in json.dumps(payload)


def test_tool_without_names_covers_the_chat_scope(client):
    payload, _ = run(client, "get_library_structure",
                     doc_ids=["pi-2022", "pi-2025"])
    assert sorted(d["doc_name"] for d in payload["documents"]) == [
        "FY2022.pdf", "FY2025.pdf"]
    payload, _ = run(client, "get_library_structure")
    assert len(payload["documents"]) == 3


def test_tool_refuses_unknown_and_out_of_scope_names(client):
    payload, is_error = run(client, "get_library_structure",
                            doc_names=["FY2022.pdf", "FY2O25.pdf"])
    assert is_error and "FY2O25.pdf" in json.dumps(payload)
    assert "FY2025.pdf" in json.dumps(payload)  # similar_files
    payload, is_error = run(client, "get_library_structure",
                            doc_ids=["pi-2022"], doc_names=["other.pdf"])
    assert is_error
    for bad in ([], [""], "FY2022.pdf"):
        _, is_error = run(client, "get_library_structure", doc_names=bad)
        assert is_error


def test_tool_paginates_a_large_library(client, monkeypatch):
    monkeypatch.setattr(agent_tools, "_CHAR_BUDGET", 900)
    seen, part = [], 1
    while True:
        payload, is_error = run(client, "get_library_structure", part=part)
        assert not is_error
        seen += [d["doc_name"] for d in payload["documents"]]
        if not payload.get("pagination", {}).get("has_more"):
            break
        part += 1
    assert part > 1
    assert set(seen) == {"FY2022.pdf", "FY2025.pdf", "other.pdf"}


def test_tool_and_guidance_follow_library_view(client):
    names = [tool.__name__ for tool in client.agent_tools()]
    assert "get_library_structure" in names
    assert names == list(tool_names())
    assert "MULTI-DOCUMENT QUESTIONS:" in agent_tools._base_instructions(client)
    client.library_view = False
    assert "get_library_structure" not in [
        tool.__name__ for tool in client.agent_tools()]
    assert "get_library_structure" not in agent_tools._base_instructions(client)


def test_tool_is_local_only():
    cloud = PageIndexClient(api_key="k")
    payload, is_error = run(cloud, "get_library_structure")
    assert is_error and "local-only" in json.dumps(payload)
    assert agent_tools._library_enabled(cloud) is False


# (chat over a doc_id list: test_local_chat.test_chat_on_a_doc_list_*)


# ── topic index ──

def fake_topics(prompt):
    keys = {}
    for line in prompt.splitlines():
        parts = line.split(" | ")
        if len(parts) == 5 and parts[0].isdigit():
            keys.setdefault("item one" in parts[4] and "Item 1" or parts[2],
                            []).append(parts[0])
    item_keys = [key for title, ks in keys.items() if title == "Item 1"
                 for key in ks]
    return json.dumps({"topics": {"manufacturing partners": item_keys,
                                  "bogus": ["999", "x"]}})


def test_build_topic_index_maps_topics_to_sections():
    roots = [{"doc_name": "a.pdf", "doc_id": "pi-a", "nodes": make_tree("a")},
             {"doc_name": "b.pdf", "doc_id": "pi-b", "nodes": make_tree("b")}]
    index = library.build_topic_index(roots, fake_topics, model="m",
                                      created_at={"pi-a": "t1", "pi-b": "t2"})
    assert index["version"] == library.TOPIC_INDEX_VERSION
    assert index["topics"] == {"manufacturing partners": [
        {"doc_id": "pi-a", "node_id": "0001", "start_index": 2, "end_index": 4},
        {"doc_id": "pi-b", "node_id": "0001", "start_index": 2, "end_index": 4},
    ]}
    assert "bogus" not in index["topics"]


def test_topic_batches_reuse_earlier_topic_names(monkeypatch):
    monkeypatch.setattr(library, "_TOPIC_BATCH_CHARS", 60)
    prompts = []

    def complete(prompt):
        prompts.append(prompt)
        return '{"topics": {"segments": ["0"]}}'

    roots = [{"doc_name": "a.pdf", "doc_id": "pi-a", "nodes": make_tree("a")}]
    library.build_topic_index(roots, complete)
    assert len(prompts) > 1
    assert "Reuse these existing topic names" not in prompts[0]
    assert '["segments"]' in prompts[1]


def test_client_builds_stores_and_shows_topics(client, store, monkeypatch):
    import pageindex.utils as utils
    calls = []

    def fake_completion(model, prompt, **kwargs):
        calls.append(model)
        return fake_topics(prompt)

    monkeypatch.setattr(utils, "llm_completion", fake_completion)
    result = client.build_library_topics()
    assert result["documents"] == 3 and result["topics"] == 1
    assert calls and set(calls) == {client.model}
    assert store.get_library_index()["topics"]["manufacturing partners"]

    # off by default: neither the view nor the tool shows it
    assert "topic index" not in client.document_context(["pi-2022", "pi-2025"])
    client.library_topics = True
    block = client.document_context(["pi-2022", "pi-2025"])
    lines = block.splitlines()
    topics = json.loads(lines[lines.index(
        "Library topic index (topic → sections):") + 1])
    assert list(topics) == ["manufacturing partners"]
    assert sorted(topics["manufacturing partners"],
                  key=lambda ref: ref["doc_name"]) == [
        {"doc_name": "FY2022.pdf", "node_id": "0001", "pages": "2-4"},
        {"doc_name": "FY2025.pdf", "node_id": "0001", "pages": "2-4"}]
    payload, _ = run(client, "get_library_structure", doc_ids=["pi-2025"])
    assert payload["topics"] == {
        "manufacturing partners": [
            {"doc_name": "FY2025.pdf", "node_id": "0001", "pages": "2-4"}]}

    # a deleted or replaced document drops out of the stored index
    client.delete_document("pi-2022")
    seed(store, "pi-2025", "FY2025.pdf", "2026-09-01T00:00:00.000")
    payload, _ = run(client, "get_library_structure")
    assert payload["topics"] == {
        "manufacturing partners": [
            {"doc_name": "other.pdf", "node_id": "0001", "pages": "2-4"}]}


def test_build_library_topics_is_local_only():
    with pytest.raises(PageIndexAPIError, match="local-only"):
        PageIndexClient(api_key="k").build_library_topics()
