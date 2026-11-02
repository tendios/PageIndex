"""Bounded search with real stores/public client APIs and a scripted API model.
No network, no modifications of the ordinary chat or evidence paths.
"""
import copy
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from pageindex import MultiDocumentOptions, MultiDocumentSearch, PageIndexClient
from pageindex import multidoc
from pageindex.stores import FileDocumentStore


def seed(store, doc_id, text=None, *, nodes=2, name=None):
    pages = [{"page_index": i + 1, "markdown": text or f"{doc_id} requirement {i + 1}: 300 euros, except on Sundays."}
             for i in range(2)]
    tree = [{"node_id": str(i), "title": f"Clause {i}", "summary": f"Requirements {doc_id}",
             "start_index": 1 + i % 2, "end_index": 1 + i % 2} for i in range(nodes)]
    store.save_document(doc_id, {"id": doc_id, "name": name or f"{doc_id}.pdf", "status": "completed",
                                "pageNum": 2, "createdAt": "fixed", "metadata": {}}, tree, pages)


@pytest.fixture(params=["files", "sql"])
def client(request, tmp_path):
    if request.param == "sql":
        pytest.importorskip("sqlalchemy")
        from pageindex.stores import SqlDocumentStore
        store = SqlDocumentStore(f"sqlite:///{tmp_path / 'docs.db'}")
    else:
        store = FileDocumentStore(str(tmp_path))
    for i in range(10):
        seed(store, f"doc-{i}")
    return PageIndexClient(index={"store": store}, chat="openai/test")


def envelope(data, finish="stop", usage=True):
    return {"choices": [{"finish_reason": finish,
                         "message": {"content": json.dumps(data, ensure_ascii=False)}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 25, "total_tokens": 125} if usage else None}


def script(**kwargs):
    body = json.loads(kwargs["messages"][-1]["content"])
    rows = body["records"]
    if body["phase"] == "structure":
        return envelope({"read": [{"ref": r["ref"], "pages": [r["start_page"]]} for r in rows
                                  if r.get("start_page") is not None]})
    if body["phase"] == "extract":
        return envelope({"findings": [{"source_id": r["source_id"], "quote": r["text"], "note": "Relevant exception"}
                                     for r in rows if r["text"].strip()]})
    return envelope({"answer": " ".join(f'{r["quote"]} [{r["id"]}]' for r in rows) or "No supported finding."})


@pytest.fixture(autouse=True)
def transport(monkeypatch):
    monkeypatch.setattr(multidoc, "_complete", script)


def query(client, **kwargs):
    return MultiDocumentSearch(client, **kwargs).query("What are the requirements and exceptions?",
                                                       doc_ids=[f"doc-{i}" for i in range(10)])


def test_complete_run_covers_ten_documents_and_preserves_sources(client):
    before = set(vars(client))
    result = query(client)
    assert result["status"] == "complete", result["reasons"]
    assert result["contract"] == "pageindex.multidoc/1"
    assert len(result["documents"]) == 10 and len(result["citations"]) == 20
    assert all(d["structure_complete"] and d["all_pages_read"] for d in result["documents"])
    assert "doc-9 requirement 2" in result["answer"]
    assert all(c["validation"] == "literal" and len(c["fingerprint"]) == 64 for c in result["citations"])
    assert result["usage"]["total_tokens"] == 125 * len(result["calls"])
    assert result["usage_complete"] and result["pending_stage"] is None
    assert all("messages" not in c for c in result["calls"])
    # No class/client toggles leaked into future ordinary chat requests.
    assert set(vars(client)) == before
    assert client.library_view is True and client.library_topics is False
    assert client.get_document("doc-0")["metadata"] == {}


def test_opt_in_does_not_change_single_or_multi_chat_context_or_tools(client):
    contexts = [client.document_context(ids) for ids in ("doc-0", ["doc-0"], ["doc-0", "doc-9"])]
    tools = [t.__name__ for t in client.agent_tools()]
    query(client)
    assert contexts == [client.document_context(ids) for ids in ("doc-0", ["doc-0"], ["doc-0", "doc-9"])]
    assert tools == [t.__name__ for t in client.agent_tools()]
    client.library_view = False
    before = client.document_context(["doc-0", "doc-9"])
    query(client)
    assert client.document_context(["doc-0", "doc-9"]) == before
    assert client.library_view is False


def test_reject_single_document_without_calling_chat(client, monkeypatch):
    monkeypatch.setattr(multidoc, "_complete", lambda **kw: pytest.fail("Model called"))
    search = MultiDocumentSearch(client)
    for ids in (["doc-0"], ["doc-0", "doc-0"], [], "doc-0", [None, "doc-1"]):
        with pytest.raises(ValueError):
            search.query("q", doc_ids=ids)
    with pytest.raises(ValueError):
        MultiDocumentSearch(PageIndexClient(api_key="key", chat="openai/test"))
    with pytest.raises(ValueError):
        MultiDocumentSearch(client, model="claude-code/test")


def test_missing_document_fails_before_any_model_request(client):
    result = MultiDocumentSearch(client).query("q", doc_ids=["doc-0", "missing"])
    assert result["status"] == "failed" and result["calls"] == []
    assert result["reasons"][0]["code"] == "source_error"


@pytest.mark.parametrize("finish", ["length", "content_filter", "tool_calls", None])
def test_incomplete_provider_response_is_not_complete(client, monkeypatch, finish):
    monkeypatch.setattr(multidoc, "_complete", lambda **kw: envelope({"read": []}, finish=finish))
    result = query(client)
    assert result["status"] == "failed"
    assert result["reasons"][0]["code"] == "incomplete_answer"
    assert result["usage"]["total_tokens"] == 125 * len(result["calls"])
    assert result["answer"] == ""


def test_limits_return_work_pending_not_silent_success(client):
    result = query(client, options=MultiDocumentOptions(max_calls=1))
    assert result["status"] == "partial"
    assert result["reasons"][0]["code"] == "max_calls"
    assert any(d["pages_pending"] for d in result["documents"])
    assert not any(d["all_pages_read"] for d in result["documents"])
    result = query(client, options=MultiDocumentOptions(max_total_tokens=100))
    assert result["status"] == "failed" and result["calls"] == []
    assert result["reasons"][0]["code"] == "max_total_tokens"


def test_provider_timeout_is_bounded_and_no_hidden_retries(client, monkeypatch):
    def fail(**kw):
        assert 0 < kw["timeout"] <= 3
        assert kw["num_retries"] == 0 and kw["stream"] is False
        raise TimeoutError("slow provider")
    monkeypatch.setattr(multidoc, "_complete", fail)
    result = query(client, options=MultiDocumentOptions(timeout_seconds=3))
    assert result["status"] == "failed" and len(result["calls"]) == 1
    assert not result["usage_complete"] and result["budget_tokens_reserved"] > 0


def test_out_of_scope_read_and_nonliteral_quote_are_rejected(client, monkeypatch):
    monkeypatch.setattr(multidoc, "_complete", lambda **kw: envelope({"read": [{"ref": "other/doc", "pages": [1]}]}))
    result = query(client)
    assert result["reasons"][0]["code"] == "invalid_reference"
    def fabricated(**kw):
        body = json.loads(kw["messages"][-1]["content"])
        if body["phase"] == "extract":
            return envelope({"findings": [{"source_id": body["records"][0]["source_id"],
                                          "quote": "fabricated quotation", "note": "x"}]})
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", fabricated)
    result = query(client)
    # A fabricated quote drops that finding alone: the run completes, nothing is cited, the drop is on record.
    assert result["status"] == "complete" and result["citations"] == [] and result["reasons"] == []
    extract_calls = [c for c in result["calls"] if c["stage"] == "extract"]
    assert len(result["findings_dropped"]) == len(extract_calls) == sum(c["findings_dropped"] for c in extract_calls)
    assert all(d["reason"] == "invalid_quote" and d["quote"] == "fabricated quotation" and d["doc_id"] and d["page"]
               for d in result["findings_dropped"])


def test_source_replacement_even_with_same_timestamp_and_tree_is_detected(client, monkeypatch):
    changed = False
    def replace_before_synthesis(**kw):
        nonlocal changed
        if not changed and json.loads(kw["messages"][-1]["content"])["phase"] == "synthesis":
            # Neither a service _revision nor createdAt changed.
            seed(client._api._store, "doc-0", "Replacement source text")
            changed = True
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", replace_before_synthesis)
    result = query(client)
    assert result["status"] == "partial" and result["answer"] == "" and result["answer_parts"] == []
    assert result["reasons"][-1]["code"] == "source_changed"
    assert result["documents"][0]["status"] == "stale"
    assert not any(c["doc_id"] == "doc-0" for c in result["citations"])


def test_parallel_queries_share_no_mutable_run_state(client):
    search = MultiDocumentSearch(client)
    with ThreadPoolExecutor(max_workers=2) as pool:
        a, b = list(pool.map(lambda ids: search.query("requirements?", doc_ids=ids),
                            [["doc-0", "doc-1"], ["doc-8", "doc-9"]]))
    assert a["status"] == b["status"] == "complete"
    assert {c["doc_id"] for c in a["citations"]} == {"doc-0", "doc-1"}
    assert {c["doc_id"] for c in b["citations"]} == {"doc-8", "doc-9"}


def test_large_outlines_are_paginated_and_all_fragments_preserved(client, monkeypatch):
    # Fast conservative test token counter; the transport still sees real data.
    monkeypatch.setattr(multidoc, "_count", lambda messages, model: len(json.dumps(messages)) // 4 + 1)
    seed(client._api._store, "doc-0", nodes=50)
    snapshot = multidoc._snapshot(client, "doc-0")
    snapshot["tree"][0]["summary"] = "deep exception " * 1000
    client._api._store.save_document("doc-0", snapshot["document"], snapshot["tree"], snapshot["pages"])
    opts = MultiDocumentOptions(max_input_tokens=1500, max_total_tokens=500000)
    result = MultiDocumentSearch(client, options=opts).query("exceptions?", doc_ids=["doc-0", "doc-9"], include_trace=True)
    assert result["status"] == "complete", result["reasons"]
    assert all(d["structure_complete"] for d in result["documents"])
    structure = [c for c in result["calls"] if c["stage"] == "structure"]
    assert len(structure) > 1
    rows = [r for c in structure for r in json.loads(c["messages"][-1]["content"])["records"]]
    pieces = [r for r in rows if r["ref"] == "D0001/0"]
    reconstructed = json.loads("".join(p["serialized_fragment"] for p in pieces))
    assert reconstructed["summary"] == "deep exception " * 1000
    assert any(r["doc_id"] == "doc-9" for r in json.loads(structure[0]["messages"][-1]["content"])["records"])
    assert all(c["input_tokens_estimated"] <= opts.max_input_tokens for c in result["calls"])


def test_split_page_only_counted_read_after_all_fragments(client, monkeypatch):
    monkeypatch.setattr(multidoc, "_count", lambda messages, model: len(json.dumps(messages)) // 4 + 1)
    seed(client._api._store, "doc-0", text="Large original passage. " * 2000)
    opts = MultiDocumentOptions(max_input_tokens=1500, max_calls=2)
    result = MultiDocumentSearch(client, options=opts).query("q", doc_ids=["doc-0", "doc-9"])
    assert result["status"] == "partial"
    assert result["documents"][0]["pages_read"] == []
    assert result["documents"][0]["pages_pending"] == [1, 2]


def test_synthesis_overflow_keeps_parts_and_is_explicitly_partial(client, monkeypatch):
    monkeypatch.setattr(multidoc, "_count", lambda messages, model: len(json.dumps(messages)) // 4 + 1)
    for i in range(10):
        seed(client._api._store, f"doc-{i}", text=f"Requirement {i}. " * 75)
    result = query(client, options=MultiDocumentOptions(max_input_tokens=1500))
    assert result["status"] == "partial" and result["answer"] == ""
    assert result["reasons"][0]["code"] == "synthesis_partitioned"
    assert len(result["answer_parts"]) > 1
    assert len({c["doc_id"] for c in result["citations"]}) == 10


def test_trace_opt_in_and_unknown_usage(client, monkeypatch):
    def no_usage(**kw):
        response = script(**kw)
        response["usage"] = None
        return response
    monkeypatch.setattr(multidoc, "_complete", no_usage)
    result = MultiDocumentSearch(client).query("q", doc_ids=["doc-0", "doc-9"], include_trace=True)
    assert result["status"] == "complete" and not result["usage_complete"]
    assert all(c.get("messages") and c.get("response") for c in result["calls"])


@pytest.mark.parametrize("changes", [dict(max_calls=0), dict(max_input_tokens=True), dict(timeout_seconds=float("inf")),
                                    dict(max_input_tokens=32000), dict(passage_overlap_chars=-1), dict(token_margin=-1)])
def test_invalid_options(changes):
    with pytest.raises(ValueError):
        MultiDocumentOptions(**changes)


def test_backend_cannot_override_budget_or_messages(client):
    for key in ("messages", "model", "tools", "max_tokens", "extra_body", "fallbacks"):
        with pytest.raises(ValueError):
            MultiDocumentSearch(client, backend={key: "bad"})


def test_structure_pagination_and_branch_expansion_are_model_free(client, monkeypatch):
    monkeypatch.setattr(multidoc, "_complete", lambda **kw: pytest.fail("Navigation called a model"))
    monkeypatch.setattr(multidoc, "_count", lambda messages, model: len(json.dumps(messages)) // 4 + 1)
    seed(client._api._store, "doc-0", nodes=35)
    search = MultiDocumentSearch(client, options=MultiDocumentOptions(max_input_tokens=1500))
    ids = ["doc-0", "doc-9"]
    first = search.structure_page(doc_ids=ids)
    assert first["has_more"]
    rows, part = list(first["records"]), first
    while part["has_more"]:
        part = search.structure_page(doc_ids=ids, cursor=part["next_cursor"])
        rows.extend(part["records"])
    assert len(rows) == 39 and len({r["cursor"] for r in rows}) == 39
    assert {r["kind"] for r in rows} == {"document", "section"}
    assert all(r["children"] == 0 for r in rows if r["kind"] == "section")
    branch = search.structure_page(doc_ids=ids, root={"doc_id": "doc-0", "node_id": "17"})
    assert [r["node_id"] for r in branch["records"]] == ["17"]
    with pytest.raises(ValueError):
        search.structure_page(doc_ids=ids, root={"doc_id": "doc-1", "node_id": "0"})
    with pytest.raises(ValueError, match="mismatched"):
        search.structure_page(doc_ids=ids[::-1], cursor=first["next_cursor"])
    seed(client._api._store, "doc-0", "Replaced sources")
    with pytest.raises(ValueError, match="Stale"):
        search.structure_page(doc_ids=ids, cursor=first["next_cursor"])


def test_cut_structure_batch_is_split_with_all_attempts_accounted(client, monkeypatch):
    def cut_large(**kw):
        body = json.loads(kw["messages"][-1]["content"])
        if body["phase"] == "structure" and len(body["records"]) > 6:
            return envelope({"read": []}, finish="length")
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", cut_large)
    result = query(client)
    assert result["status"] == "complete", result["reasons"]
    assert all(d["structure_complete"] for d in result["documents"])
    assert any(c.get("recovery") == "split_batch" for c in result["calls"])
    assert result["usage"]["total_tokens"] == 125 * len(result["calls"])


def test_verified_quotes_keep_real_sdk_geometry_and_multiple_matches(client):
    from pageindex.pdf_extraction import _page_record, parser_config
    store = client._api._store
    word = {"text": "Repeated quote Repeated quote", "x": 20, "y": 30, "width": 150, "height": 10}
    raw = {"page_num": 1, "width": 200, "height": 300, "text": word["text"],
           "text_items": [{**word, "words": [word]}], "blocks": [{"kind": "paragraph", "text": word["text"], "bbox": word}]}
    page = _page_record(raw, parser_config("liteparse"), False)
    store.save_document("doc-0", store.get_meta("doc-0"), store.get_tree("doc-0"),
                        [page, {"page_index": 2, "markdown": "Other page"}])
    result = MultiDocumentSearch(client).query("q", doc_ids=["doc-0", "doc-9"])
    assert result["status"] == "complete"
    cite = next(c for c in result["citations"] if c["doc_id"] == "doc-0" and c["page"] == 1)
    assert cite["locations"] == client.get_text_locations("doc-0", cite["quote"], pages="1")
    assert cite["locations"][0]["boxes"]


def test_cited_ids_tolerate_annotations_inside_brackets(client, monkeypatch):
    def annotated(**kw):
        body = json.loads(kw["messages"][-1]["content"])
        if body["phase"] == "synthesis":
            rows = body["records"]
            return envelope({"answer": f'{rows[0]["quote"]} [{rows[0]["id"]}, pág. {rows[0]["page"]}] '
                                       f'y [{rows[1]["id"]}][{rows[2]["id"]}]. Sin fuente [pág. 9].'})
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", annotated)
    result = MultiDocumentSearch(client).query("q", doc_ids=["doc-0", "doc-9"])
    assert result["status"] == "complete", result["reasons"]
    assert result["answer_parts"][0]["finding_ids"] == ["E000001", "E000002", "E000003"]
    assert multidoc.cited_ids(result["answer"]) == ["E000001", "E000002", "E000003"]
    assert multidoc.cited_ids("[E000001-E000002] [E7] E000003 [x]") == ["E000001", "E000002"]
    assert multidoc.cited_ids("") == []


def test_bad_extraction_does_not_erase_actual_page_read_coverage(client, monkeypatch):
    def bad_json(**kw):
        if json.loads(kw["messages"][-1]["content"])["phase"] == "extract":
            return {"choices": [{"finish_reason": "stop", "message": {"content": "not json"}}]}
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", bad_json)
    result = query(client)
    assert result["status"] == "partial" and result["citations"] == []
    assert all(d["pages_read"] == [1, 2] and d["extraction_pending"] for d in result["documents"])


def test_fit_tolerates_formatting_but_never_letters_or_figures():
    text = "| Plazo de ejecución | 36 meses |\n| --- | --- |\n| Precio | 2.180.437,62 € «IVA excluido» |"
    assert multidoc._fit("Plazo de ejecución | 36 meses", text) == ("Plazo de ejecución | 36 meses", "exact_normalized")
    # Table bars, case, accents, punctuation and quote marks are formatting: the passage's own text is cited.
    assert multidoc._fit("PLAZO DE EJECUCION: 36 meses", text) == ("Plazo de ejecución | 36 meses", "snapped")
    assert multidoc._fit('Precio 2.180.437,62 € "IVA excluido"', text) == ("Precio | 2.180.437,62 € «IVA excluido", "snapped")
    # A figure, a number separator or a word is content: no fit.
    assert multidoc._fit("Plazo de ejecución 30 meses", text) is None
    assert multidoc._fit("Precio 2180437,62", text) is None
    assert multidoc._fit("IVA incluido", text) is None
    assert multidoc._fit("", text) is None and multidoc._fit("***", text) is None and multidoc._fit(None, text) is None
    assert multidoc._plain("| Plazo de ejecución | 36 meses |\n| --- | --- |\n| a\\|b<br>c | d |") == "Plazo de ejecución 36 meses a|b c d"


def test_formatting_slips_snap_to_the_stored_text_and_keep_geometry(client, monkeypatch):
    """A quote without the table's bars, in capitals and without accents is a formatting slip, not a fabrication: the
    citation carries the passage's own characters, says so in ``match``, and its boxes come from the page's words."""
    from pageindex.pdf_extraction import _page_record, parser_config
    store = client._api._store
    cells = [{"text": "Plazo de ejecución", "x": 20, "y": 30, "width": 100, "height": 10},
             {"text": "36 meses", "x": 130, "y": 30, "width": 50, "height": 10}]
    raw = {"page_num": 1, "width": 200, "height": 300, "text": "Plazo de ejecución 36 meses",
           "text_items": [{**c, "words": [c]} for c in cells],
           "blocks": [{"kind": "table", "header": [{"text": c["text"]} for c in cells], "rows": [],
                       "bbox": {"x": 20, "y": 30, "width": 160, "height": 10}}]}
    page = _page_record(raw, parser_config("liteparse"), False)
    assert page["markdown"].startswith("| Plazo de ejecución | 36 meses |")
    store.save_document("doc-0", store.get_meta("doc-0"), store.get_tree("doc-0"), [page, {"page_index": 2, "markdown": "Other page"}])

    def cleaned(**kw):
        body = json.loads(kw["messages"][-1]["content"])
        if body["phase"] == "extract":
            return envelope({"findings": [{"source_id": r["source_id"], "note": "n",
                                          "quote": "PLAZO DE EJECUCION 36 meses" if r["text"].startswith("|") else r["text"]}
                                         for r in body["records"] if r["text"].strip()]})
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", cleaned)
    result = MultiDocumentSearch(client).query("q", doc_ids=["doc-0", "doc-9"])
    assert result["status"] == "complete" and result["findings_dropped"] == [], result["reasons"]
    cite = next(c for c in result["citations"] if c["doc_id"] == "doc-0" and c["page"] == 1)
    assert cite["quote"] == "Plazo de ejecución | 36 meses" and cite["match"] == "snapped" and cite["validation"] == "literal"
    assert cite["geometry_status"] == "available" and cite["locations"][0]["boxes"]
    assert cite["locations"] == client.get_text_locations("doc-0", "Plazo de ejecución 36 meses", pages="1")
    assert all(c["match"] == "exact_normalized" for c in result["citations"] if c["doc_id"] == "doc-9")
    assert cite["quote"] in result["answer"]


def test_a_quote_that_does_not_fit_drops_only_that_finding(client, monkeypatch):
    """A changed figure or word is never snapped: that finding is dropped and recorded; the batch's other findings
    are kept, the remaining batches are read and the synthesis runs."""
    def two_bad_per_batch(**kw):
        body = json.loads(kw["messages"][-1]["content"])
        if body["phase"] == "extract":
            rows = [r for r in body["records"] if r["text"].strip()]
            findings = [{"source_id": r["source_id"], "quote": r["text"], "note": "ok"} for r in rows]
            findings.append({"source_id": rows[0]["source_id"], "quote": rows[0]["text"].replace("300", "3000"), "note": "figure"})
            findings.append({"source_id": rows[0]["source_id"], "quote": rows[0]["text"].replace("except", "including"), "note": "word"})
            return envelope({"findings": findings})
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", two_bad_per_batch)
    result = query(client)
    assert result["status"] == "complete" and result["reasons"] == [] and result["pending_stage"] is None
    assert len(result["citations"]) == 20 and all(c["match"] == "exact_normalized" and "relocated_from" not in c
                                                   for c in result["citations"])
    assert "doc-9 requirement 2" in result["answer"]
    extract_calls = [c for c in result["calls"] if c["stage"] == "extract"]
    assert len(result["findings_dropped"]) == 2 * len(extract_calls) == sum(c["findings_dropped"] for c in extract_calls)
    assert {d["reason"] for d in result["findings_dropped"]} == {"invalid_quote"}
    dropped = result["findings_dropped"][0]
    assert dropped["doc_id"] == "doc-0" and dropped["page"] == 1 and dropped["source_id"].startswith("P0001-000001-")
    assert "3000 euros" in dropped["quote"]
    assert all(d["pages_read"] == [1, 2] and d["extraction_pending"] == [] for d in result["documents"])


def test_a_quote_attributed_to_the_wrong_passage_is_relocated_only_when_one_passage_holds_it(client, monkeypatch):
    def misattributed(**kw):
        body = json.loads(kw["messages"][-1]["content"])
        if body["phase"] == "extract":
            rows = [r for r in body["records"] if r["text"].strip()]
            if len(rows) < 2:
                return script(**kw)
            # The first names the second passage but quotes the first; the second names an unread passage.
            findings = [{"source_id": rows[1]["source_id"], "quote": rows[0]["text"], "note": "swapped"},
                        {"source_id": "P9999-000001-000000000", "quote": rows[1]["text"], "note": "unknown"}]
            findings += [{"source_id": r["source_id"], "quote": r["text"], "note": "ok"} for r in rows[2:]]
            return envelope({"findings": findings})
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", misattributed)
    result = query(client)
    assert result["status"] == "complete" and result["findings_dropped"] == [], result["reasons"]
    assert len(result["citations"]) == 20
    relocated = [c for c in result["citations"] if "relocated_from" in c]
    assert len(relocated) >= 2 and all(c["relocated_from"] != c["source_id"] for c in relocated)
    for c in relocated:  # the citation names the passage that actually holds the text
        assert c["quote"].startswith(c["doc_id"]) and f"requirement {c['page']}" in c["quote"]
    assert all(c["match"] == "exact_normalized" and c["validation"] == "literal" for c in result["citations"])
    assert "relocated_from" not in result["answer"] and "match" not in json.dumps(result["answer_parts"])

    # The same text in two passages of the batch: nowhere unambiguous to put it, so that finding is dropped.
    store = client._api._store
    for doc in ("doc-0", "doc-1"):
        seed(store, doc, "Identical page text in two documents.")
    def unknown_source(**kw):
        body = json.loads(kw["messages"][-1]["content"])
        if body["phase"] == "extract":
            rows = [r for r in body["records"] if r["text"].strip()]
            findings = [{"source_id": r["source_id"], "quote": r["text"], "note": "ok"} for r in rows]
            findings.append({"source_id": None, "quote": "Identical page text in two documents.", "note": "where?"})
            findings.append("not an object")
            return envelope({"findings": findings})
        return script(**kw)
    monkeypatch.setattr(multidoc, "_complete", unknown_source)
    result = MultiDocumentSearch(client).query("q", doc_ids=["doc-0", "doc-1", "doc-2"])
    assert result["status"] == "complete", result["reasons"]
    assert [d["reason"] for d in result["findings_dropped"]] == ["invalid_reference", "invalid_response"]
    assert result["findings_dropped"][0] == {"source_id": None, "doc_id": None, "page": None, "reason": "invalid_reference",
                                             "quote": "Identical page text in two documents."}
    assert result["findings_dropped"][1]["quote"] is None
