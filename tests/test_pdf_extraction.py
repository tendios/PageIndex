"""OCR routing, physical pages, and grounded local citations without paid LLMs."""
import copy
import json
import sys
from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from pageindex import PageIndexClient, PageIndexLocalClient
from pageindex.errors import PageIndexAPIError
from pageindex.pdf_extraction import PDFExtraction, _page_record, extract_liteparse, locate_text, parser_config


def page(number, text="native"):
    word = {"text": text, "x": 20, "y": 30, "width": 40, "height": 10}
    return {"page_num": number, "width": 200, "height": 300, "text": text,
            "text_items": [{**word, "words": [word]}],
            "blocks": [{"kind": "paragraph", "text": text, "bbox": word}]}


@dataclass
class Result:
    pages: list
    total_pages: int = 3
    page_errors: list = field(default_factory=list)
    text: str = ""
    doc_meta: dict | None = None


def fake_parser(monkeypatch, results, routed=(2,)):
    calls = []

    class Parser:
        def __init__(self, **kwargs):
            self.tags = bool(kwargs.get("extract_text_metadata"))
            if not self.tags:  # the light pass that reads a tagged PDF's headings is not an extraction pass
                calls.append(kwargs)

        def parse(self, data):
            if self.tags:
                raise RuntimeError("untagged fixture")
            item = results.pop(0)
            if isinstance(item, Exception):
                raise item
            return item

    monkeypatch.setitem(sys.modules, "liteparse", SimpleNamespace(LiteParse=Parser))
    monkeypatch.setitem(sys.modules, "pdf_inspector", SimpleNamespace(
        detect_pdf_bytes=lambda _: SimpleNamespace(
            pages_needing_ocr=routed, page_count=3, pdf_type="mixed", confidence=0.9, has_encoding_issues=False,
            processing_time_ms=1.5, ocr_reasons_by_page=[SimpleNamespace(page=n, reasons=["no_text"]) for n in routed])))
    monkeypatch.setattr("pageindex.pdf_extraction.version", lambda _: "test")
    return calls


def test_targeted_ocr_keeps_physical_pages_and_geometry(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    path.write_bytes(b"fixture")
    calls = fake_parser(monkeypatch, [Result([page(1), page(2, ""), page(3)]),
                                      Result([page(2, "recovered")])])
    result = extract_liteparse(str(path), "liteparse")
    assert [p["page_index"] for p in result.pages] == [1, 2, 3]
    assert result.texts == ["native", "recovered", "native"]
    assert [c["target_pages"] for c in calls] == [None, "2"]
    assert [c["ocr_enabled"] for c in calls] == [False, True]
    assert result.pages[1]["blocks"][0]["bbox"] == pytest.approx([100, 100, 300, 133.333333])
    assert result.pages[1]["layout"]["text_items"][0]["words"][0]["x"] == 20
    json.dumps(result.pages, allow_nan=False)


BLOCK_NEUTRAL = ("extract_links", "extract_annotations", "extract_form_fields", "extract_structure_tree",
                 "extract_xfa_packets", "extract_document_metadata", "include_complexity")


def test_both_passes_read_what_leaves_the_blocks_alone_and_the_caller_gets_liteparses_result(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    path.write_bytes(b"fixture")
    full = [page(1), page(2, ""), page(3)]
    for p in full:
        p.update(annotations=[{"subtype": "Link", "uri": "https://example.org"}], form_fields=[],
                 structure_tree={"roots": [{"type": "P"}]}, complexity={"is_complex": False})
    ocr = {**page(2, "recovered"), "annotations": [], "form_fields": [], "structure_tree": None, "complexity": None}
    calls = fake_parser(monkeypatch, [Result(full, text="native\n\n\n\nnative", doc_meta={"file_version": 17}),
                                      Result([ocr], doc_meta={"file_version": 0})])
    result = extract_liteparse(str(path), "liteparse")
    for call in calls:
        assert all(call[option] is True for option in BLOCK_NEUTRAL)
        assert "extract_text_metadata" not in call  # it splits the text items: other blocks
    read = result.liteparse
    document = read.pop("document")
    assert read == {"version": "test", "ocr_mode": "auto", "ocr_language": "spa", "ocr_backend": "tesseract",
                    "ocr_pages": [2], "passes": [{k: v for k, v in p.items()} for p in result.metadata["passes"]],
                    "seconds": pytest.approx(read["seconds"]),
                    "inspection": {"version": "test", "pdf_type": "mixed", "confidence": 0.9, "page_count": 3,
                                   "pages_needing_ocr": [2], "has_encoding_issues": False, "processing_time_ms": 1.5,
                                   "ocr_reasons_by_page": [{"page": 2, "reasons": ["no_text"]}]}}
    # The full pass with the OCR page in its place, its own document-level fields, the text joined again.
    assert [p["page_num"] for p in document["pages"]] == [1, 2, 3]
    assert document["pages"][1] == ocr and document["pages"][0] == full[0]
    assert document["doc_meta"] == {"file_version": 17} and document["total_pages"] == 3
    assert document["text"] == "native\n\nrecovered\n\nnative" and document["page_errors"] == []
    # The index stores the pages as before: the fields read for the caller empty, as when they were not read.
    for stored in result.pages:
        assert [stored["layout"][k] for k in ("annotations", "form_fields", "structure_tree", "complexity")] == [None] * 4
        assert stored["layout"]["blocks"] and stored["layout"]["text_items"]
    json.dumps(document, allow_nan=False)


@pytest.mark.parametrize("routed", [(), (1, 2, 3)])
def test_all_native_or_all_ocr_needs_only_one_pass(tmp_path, monkeypatch, routed):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"fixture")
    calls = fake_parser(monkeypatch, [Result([page(1), page(2), page(3)])], routed)
    result = extract_liteparse(str(path), "liteparse")
    assert len(calls) == 1
    assert calls[0]["ocr_enabled"] is bool(routed)
    assert result.metadata["ocr_pages"] == list(routed)
    # One pass: the caller's result is LiteParse's own, as it returned it.
    assert result.liteparse["document"]["pages"] == [page(1), page(2), page(3)]
    assert result.liteparse["ocr_pages"] == list(routed)


HINT_FIELDS = ("tagged_headings", "inspector_headings")


def without_hints(pages):
    return [{key: value for key, value in p.items() if key not in HINT_FIELDS} for p in pages]


def without_timings(read):
    """A read's metadata or LiteParse result without what the clock gives: its seconds, each pass's, and
    pdf-inspector's processing time."""
    read = copy.deepcopy(read)
    read.pop("seconds")
    for p in read["passes"]:
        p.pop("seconds")
    (read.get("inspection") or {}).pop("processing_time_ms", None)
    return read


def test_without_heading_hints_their_two_reads_are_left_out_and_the_rest_is_the_same(tmp_path, monkeypatch):
    path = tmp_path / "mixed.pdf"
    path.write_bytes(b"fixture")
    asked = []

    def tagged(parser_class, data, count):
        asked.append("tagged")
        return [[{"level": 1, "text": f"Tagged {n}"}] for n in range(1, count + 1)]

    def inspected(inspector, data, count):
        asked.append("inspector")
        return [[{"level": 2, "text": f"Inspected {n}"}] for n in range(1, count + 1)]

    monkeypatch.setattr("pageindex.pdf_extraction._tagged_headings", tagged)
    monkeypatch.setattr("pageindex.pdf_extraction._inspector_headings", inspected)
    reads, calls = {}, {}
    for hints in (True, False):
        asked.clear()
        calls[hints] = fake_parser(monkeypatch, [Result([page(1), page(2, ""), page(3)]),
                                                 Result([page(2, "recovered")])])
        reads[hints] = extract_liteparse(str(path), "liteparse", heading_hints=hints), list(asked)
    (with_hints, asked_with), (without, asked_without) = reads[True], reads[False]
    assert asked_with == ["tagged", "inspector"] and asked_without == []
    # The indexing's read puts them on the native pages (page 2 went to OCR); without them no page has the fields.
    assert [[key for key in HINT_FIELDS if key in p] for p in with_hints.pages] == [list(HINT_FIELDS), [],
                                                                                    list(HINT_FIELDS)]
    assert not any(key in p for p in without.pages for key in HINT_FIELDS)
    assert json.dumps(without.pages) == json.dumps(without_hints(with_hints.pages))
    assert calls[False] == calls[True] and len(calls[False]) == 2
    assert without_timings(without.liteparse) == without_timings(with_hints.liteparse)
    assert without_timings(without.metadata) == without_timings(with_hints.metadata)


@pytest.mark.parametrize("bad", [Result([page(1)]), Result([page(2), page(2)]),
                                 Result([page(2)], total_pages=2), RuntimeError("OCR unavailable")])
def test_failed_or_misnumbered_ocr_never_saves_partial_success(tmp_path, monkeypatch, bad):
    path = tmp_path / "doc.pdf"
    path.write_bytes(b"fixture")
    fake_parser(monkeypatch, [Result([page(1), page(2), page(3)]), bad])
    with pytest.raises((ValueError, RuntimeError)):
        extract_liteparse(str(path), "liteparse")


@pytest.mark.parametrize("config", ["typo", {"name": "liteparse", "ocr_mode": "typo"},
                                    {"name": "liteparse", "extra": True},
                                    {"name": "default", "ocr_mode": "auto"}])
def test_config_rejected_before_indexing(config):
    with pytest.raises(PageIndexAPIError, match="pdf_parser"):
        PageIndexClient(index={"model": "m", "pdf_parser": config})


def test_grouped_flat_and_cloud_config(tmp_path):
    for client in [PageIndexClient(index={"model": "m", "pdf_parser": "liteparse"}),
                   PageIndexLocalClient(model="m", pdf_parser="liteparse")]:
        assert client._api._pdf_parser["ocr_mode"] == "auto"
    with pytest.raises(PageIndexAPIError):
        PageIndexClient(index={"api_key": "x", "pdf_parser": "liteparse"})


def test_missing_optional_dependency_has_install_hint(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "liteparse", None)
    with pytest.raises(ImportError, match=r"pageindex\[liteparse\]"):
        extract_liteparse(str(tmp_path / "not-opened.pdf"), "liteparse")


def test_spatial_fallback_preserves_empty_columns_without_inventing_cells():
    p = page(1)
    p["text"] = "A    B    C    D\n1    2         4\n5    6    7    8\n9    0    1    2"
    p["blocks"] = [{"kind": "paragraph", "text": "A B C D 1 2 4"}]
    result = _page_record(p, parser_config("liteparse"), False)
    assert "1    2         4" in result["markdown"]
    assert result["extraction"]["text_format"] == "spatial"
    assert result["blocks"][0]["block_type"] == "page"


def test_quote_locations_are_exact_repeated_and_not_guessed():
    p = _page_record(page(2, "Almacén"), parser_config("liteparse"), True)
    words = p["layout"]["text_items"][0]["words"]
    words.extend([{**words[0], "text": "General", "x": 70},
                  {**words[0], "text": "Almacén", "y": 80},
                  {**words[0], "text": "General", "x": 70, "y": 80}])
    p["layout"]["text_items"][0]["text"] = "Almacén General Almacén General"
    matches = locate_text([p], "Almacén\n General")
    assert len(matches) == 2 and all(m["page"] == 2 for m in matches)
    assert matches[0]["boxes"][0][1] < matches[1]["boxes"][0][1]
    assert locate_text([p], "almacén principal") == []
    assert locate_text([p], "Almacén [...] General") == []
    assert locate_text([{"page_index": 1, "markdown": "Almacén General"}], "Almacén General") == []


def test_quote_locations_include_overlapping_unicode_matches():
    p = _page_record(page(1, "Ａ Ａ Ａ"), parser_config("liteparse"), False)
    matches = locate_text([p], "A\nA")
    assert len(matches) == 2
    assert all(m["text"] == "A A" and m["boxes"] for m in matches)


def test_liteparse_flows_to_flash_store_tools_and_citations(sample_pdf, tmp_path, monkeypatch):
    from pageindex import flash, utils
    from pageindex.agent_tools import _get_page_content
    pages = [_page_record(page(n, f"recovered{n}"), parser_config("liteparse"), True)
             for n in (1, 2)]
    extraction = PDFExtraction(pages, {"parser": "liteparse"})
    monkeypatch.setattr("pageindex.pdf_extraction.extract_liteparse", lambda *a: extraction)
    seen = {}

    def index(pdf, **kwargs):
        seen.update(kwargs)
        return {"structure": [{"title": "All", "start_index": 1, "end_index": 2}]}

    monkeypatch.setattr(flash, "page_index_flash", index)
    monkeypatch.setattr(utils, "generate_doc_description", lambda *a, **k: "test")
    client = PageIndexClient(index={"model": "m", "pdf_parser": "liteparse", "storage_path": str(tmp_path / "store")})
    doc = client.submit_document(sample_pdf, mode="flash")["doc_id"]
    assert seen["page_texts"] == ["recovered1", "recovered2"]
    assert "recovered2" in str(client.get_tree(doc))
    assert client.get_ocr(doc)["features"]["pdf_extraction"]["parser"] == "liteparse"
    tool, _ = _get_page_content(client, "sample.pdf", "2")
    assert "block_id=p2_text_1" in json.dumps(tool)
    assert "bbox" not in json.dumps(tool) and "text_items" not in json.dumps(tool)
    cited = '<cite doc="sample.pdf" page="2" block="p2_text_1"/>'
    assert client.get_citations(cited, doc)[0]["bbox"] == pages[1]["blocks"][0]["bbox"]
    assert "bbox" not in client.get_citations(cited.replace('page="2"', 'page="1"'), doc)[0]
    assert "bbox" not in client.get_citations(cited.replace("p2_text_1", "p2_text_999"), doc)[0]
    assert client.get_text_locations(doc, "recovered2", "2")[0]["page"] == 2
    # Reopen with the default reader: persisted layout remains usable.
    reopened = PageIndexClient(model="m", storage_path=str(tmp_path / "store"))
    assert reopened.get_block(doc, "p2_text_1")["bbox"] == pages[1]["blocks"][0]["bbox"]


def test_flash_override_reaches_optimizer_and_covers_scanned_tail(sample_pdf, monkeypatch):
    import pageindex.flash.api as api
    monkeypatch.setattr(api, "extract_toc", lambda *a, **k: {
        "structure": [{"title": "Native", "start_index": 1, "end_index": 1}],
        "page_texts": ["old", ""], "toc_source": "detected"})
    seen = {}

    async def optimize(structure, page_texts, **kwargs):
        seen.update(pages=page_texts, structure=copy.deepcopy(structure))
        return {}

    monkeypatch.setattr(api, "_optimize_and_summarize", optimize)
    result = api.page_index_flash(sample_pdf, summary_model="m", page_texts=["new", "OCR tail"])
    assert seen["pages"] == ["new", "OCR tail"]
    assert any(n["start_index"] == n["end_index"] == 2 for n in result["structure"])
    with pytest.raises(ValueError, match="count"):
        api.page_index_flash(sample_pdf, page_texts=["wrong count"])


def test_large_scan_falls_back_to_standard_with_ocr_pages(tmp_path, monkeypatch):
    from pageindex.local_api import LocalAPI
    import pageindex.flash as flash
    api = LocalAPI(str(tmp_path), "m", "m", pdf_parser="liteparse")
    extraction = PDFExtraction([{"markdown": f"OCR page {i}"} for i in range(12)], {})
    nodes = [{"title": f"Page {i}", "start_index": i, "end_index": i} for i in range(1, 13)]
    monkeypatch.setattr(flash, "page_index_flash", lambda *a, **k: {"toc_source": "pages", "structure": nodes})
    seen = []
    monkeypatch.setattr(api, "_index_standard", lambda file, texts: (seen.extend(texts) or nodes, "OCR"))
    assert api._index_flash("scan.pdf", extraction)[1] == "OCR"
    assert seen == extraction.texts
    assert extraction.metadata["structure_backend"] == "standard_fallback"
    configured = LocalAPI(str(tmp_path), "m", "m", pdf_parser="liteparse", summary_max_words=50)
    with pytest.raises(PageIndexAPIError, match="cannot honor"):
        configured._index_flash("scan.pdf", extraction)


def test_standard_receives_same_extracted_text_as_stored_pages(sample_pdf, tmp_path, monkeypatch):
    from pageindex import page_index_classic as classic
    extraction = PDFExtraction([{"page_index": 1, "markdown": "OCR first"},
                                {"page_index": 2, "markdown": "OCR second"}], {})
    monkeypatch.setattr("pageindex.pdf_extraction.extract_liteparse", lambda *a: extraction)
    seen = {}

    def standard(file, options, **kwargs):
        seen.update(kwargs)
        return {"structure": [{"title": "Scan", "start_index": 1, "end_index": 2}]}

    monkeypatch.setattr(classic, "page_index_main", standard)
    client = PageIndexLocalClient(model="m", pdf_parser="liteparse", storage_path=str(tmp_path / "store"))
    doc = client.submit_document(sample_pdf, mode="standard")["doc_id"]
    assert [p[0] for p in seen["page_list"]] == extraction.texts
    assert client.get_page_content(doc, "2")[0]["markdown"] == "OCR second"


def test_real_liteparse_rotated_word_geometry(tmp_path):
    pytest.importorskip("liteparse")
    pytest.importorskip("pdf_inspector")
    from io import BytesIO
    from PyPDF2 import PdfReader, PdfWriter
    from conftest import build_pdf
    reader = PdfReader(BytesIO(build_pdf(["Hello rotated world"])))
    writer = PdfWriter()
    writer.add_page(reader.pages[0].rotate(90))
    path = tmp_path / "rotated.pdf"
    with path.open("wb") as f:
        writer.write(f)
    result = extract_liteparse(str(path), {"name": "liteparse", "ocr_mode": "disabled"})
    layout = result.pages[0]["layout"]
    assert layout["width"] == pytest.approx(792)
    assert layout["height"] == pytest.approx(612)
    matches = locate_text(result.pages, "Hello")
    assert matches
    box = matches[0]["boxes_points"][0]
    # The original baseline near (72,720) becomes right/top after /Rotate90.
    assert 700 < box["x"] < 750 and 60 < box["y"] < 100


def test_submit_returns_liteparses_result_only_when_asked_and_never_stores_it(sample_pdf, tmp_path, monkeypatch):
    from pageindex import flash, utils
    pages = [_page_record(page(n, f"text{n}"), parser_config("liteparse"), False) for n in (1, 2)]
    document = {"pages": [page(1, "text1"), page(2, "text2")], "text": "text1\n\ntext2", "total_pages": 2,
                "doc_meta": {"xmp": "<x:xmpmeta/>"}, "xfa_packets": [{"index": 0, "content": "<xdp/>"}]}
    monkeypatch.setattr("pageindex.pdf_extraction.extract_liteparse", lambda *a: PDFExtraction(
        pages, {"parser": "liteparse"}, liteparse={"version": "test", "document": document}))
    monkeypatch.setattr(flash, "page_index_flash",
                        lambda pdf, **k: {"structure": [{"title": "All", "start_index": 1, "end_index": 2}]})
    monkeypatch.setattr(utils, "generate_doc_description", lambda *a, **k: "test")
    store = tmp_path / "store"
    client = PageIndexClient(index={"model": "m", "pdf_parser": "liteparse", "storage_path": str(store)})
    asked = client.submit_document(sample_pdf, mode="flash", doc_id="asked", return_liteparse=True)
    assert asked["liteparse"] == {"version": "test", "document": document} and asked["timings"]["total"] >= 0
    plain = client.submit_document(sample_pdf, mode="flash", doc_id="plain", if_exists="replace")
    assert "liteparse" not in plain
    stored = "".join(f.read_text() for f in store.rglob("*.json"))
    assert "xmpmeta" not in stored and "<xdp/>" not in stored
    with pytest.raises(PageIndexAPIError, match="return_liteparse needs") as error:
        PageIndexClient(model="m", storage_path=str(tmp_path / "other")).submit_document(
            sample_pdf, return_liteparse=True)
    assert error.value.status_code == 400
    with pytest.raises(PageIndexAPIError, match="return_liteparse is local-only"):
        PageIndexClient(api_key="x").submit_document(sample_pdf, return_liteparse=True)


def test_real_liteparse_reads_the_document_level_fields_for_the_caller(tmp_path):
    pytest.importorskip("liteparse")
    pytest.importorskip("pdf_inspector")
    from conftest import build_pdf
    path = tmp_path / "plain.pdf"
    path.write_bytes(build_pdf(["Hello page one", "Hello page two"]))
    result = extract_liteparse(str(path), "liteparse")
    document = result.liteparse["document"]
    assert document["total_pages"] == 2 and [p["page_num"] for p in document["pages"]] == [1, 2]
    assert document["doc_meta"] is not None and document["xfa_packets"] is not None
    assert all(p["complexity"] is not None and p["annotations"] is not None for p in document["pages"])
    assert result.liteparse["inspection"]["page_count"] == 2
    # The blocks the caller gets are the ones the index stores.
    assert [p["blocks"] for p in document["pages"]] == [p["layout"]["blocks"] for p in result.pages]
    json.dumps(result.liteparse, allow_nan=False)


def test_real_liteparse_without_heading_hints_reads_once_and_gives_the_same_pages(tmp_path, monkeypatch):
    liteparse = pytest.importorskip("liteparse")
    pdf_inspector = pytest.importorskip("pdf_inspector")
    from conftest import build_pdf
    path = tmp_path / "titled.pdf"
    path.write_bytes(build_pdf([
        [(72, 720, 24, "1. Objeto del contrato"), (72, 690, 11, "El contrato tiene por objeto el suministro."),
         (72, 670, 11, "Se entrega en el Almacen General cada mes.")],
        [(72, 720, 24, "2. Precio"), (72, 690, 11, "El precio es de mil euros por cada lote.")]]))
    with_hints = extract_liteparse(str(path), "liteparse")
    assert any("inspector_headings" in p for p in with_hints.pages)  # pdf-inspector names a title here
    built, markdown_reads = [], []

    class Recorded(liteparse.LiteParse):
        def __init__(self, **options):
            built.append(options)
            super().__init__(**options)

    monkeypatch.setattr(liteparse, "LiteParse", Recorded)
    monkeypatch.setattr(pdf_inspector, "extract_pages_markdown_bytes", markdown_reads.append)
    without = extract_liteparse(str(path), "liteparse", heading_hints=False)
    # One LiteParse read, the indexing's (no OCR here); no read with the text metadata, no pdf-inspector markdown.
    assert len(built) == 1 and not any(options.get("extract_text_metadata") for options in built)
    assert markdown_reads == []
    assert json.dumps(without.pages) == json.dumps(without_hints(with_hints.pages))
    assert json.dumps(without.liteparse["document"]) == json.dumps(with_hints.liteparse["document"])
    assert without_timings(without.liteparse) == without_timings(with_hints.liteparse)


def test_real_liteparse_ocr_of_image_only_pdf(tmp_path):
    pytest.importorskip("liteparse")
    pytest.importorskip("pdf_inspector")
    from PIL import Image, ImageDraw, ImageFont
    from PyPDF2 import PdfReader

    # A generated fixture with no embedded text and no system-font dependency.
    image = Image.new("RGB", (500, 160), "white")
    ImageDraw.Draw(image).text((35, 60), "Entrega en Almacen General",
                              font=ImageFont.load_default(), fill="black")
    path = tmp_path / "scanned.pdf"
    image.resize((2000, 640)).save(path, "PDF", resolution=200)
    assert not (PdfReader(str(path)).pages[0].extract_text() or "").strip()
    result = extract_liteparse(str(path), "liteparse")
    assert result.metadata["ocr_pages"] == [1]
    assert len(result.pages) == 1
    assert "General" in result.texts[0]
    matches = locate_text(result.pages, "General")
    assert matches and matches[0]["boxes"]
    assert all(0 <= value <= 1000 for box in matches[0]["boxes"] for value in box)


def test_default_reader_keeps_legacy_payload_without_optional_packages(sample_pdf, tmp_path, monkeypatch):
    from pageindex import flash, utils
    monkeypatch.setitem(sys.modules, "liteparse", None)
    monkeypatch.setitem(sys.modules, "pdf_inspector", None)
    seen = {}

    def index(pdf, **kwargs):
        seen.update(kwargs)
        return {"structure": [{"title": "All", "start_index": 1, "end_index": 2}]}

    monkeypatch.setattr(flash, "page_index_flash", index)
    monkeypatch.setattr(utils, "generate_doc_description", lambda *a, **k: "test")
    client = PageIndexClient(model="m", storage_path=str(tmp_path / "store"))
    doc = client.submit_document(sample_pdf)["doc_id"]
    assert "page_texts" not in seen
    result = client.get_ocr(doc)
    assert result["features"] == {}
    assert all(set(p) == {"page_index", "markdown"} for p in result["result"])
    assert "Hello page one" in result["result"][0]["markdown"]
    assert client.get_text_locations(doc, "Hello") == []


def test_failed_ocr_does_not_replace_existing_index(sample_pdf, tmp_path, monkeypatch):
    from pageindex.stores import FileDocumentStore
    from test_stores import make_meta, TREE, PAGES
    store = FileDocumentStore(str(tmp_path / "store"))
    original = make_meta("existing")
    store.save_document("existing", original, TREE, PAGES)

    def failed(*args):
        raise RuntimeError("OCR unavailable")

    monkeypatch.setattr("pageindex.pdf_extraction.extract_liteparse", failed)
    client = PageIndexClient(index={"model": "m", "store": store, "pdf_parser": "liteparse"})
    with pytest.raises(PageIndexAPIError, match="OCR unavailable"):
        client.submit_document(sample_pdf, doc_id="existing", if_exists="replace")
    assert store.get_meta("existing") == original
    assert store.get_tree("existing") == TREE
    assert store.get_pages("existing") == PAGES
