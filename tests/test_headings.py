"""Mode headings: the tree from LiteParse's headings, the document's numbering, its index pages, the layout titles
and a judge for the rest; summaries and description as flash."""
from __future__ import annotations

import json

import pytest

from pageindex import PageIndexClient
from pageindex.errors import PageIndexAPIError
from pageindex.headings import TypesafeHeadingJudge, build_headings_structure, collect
from pageindex.headings.builder import build_structure
from pageindex.headings.candidates import Candidate
from pageindex.pdf_extraction import PDFExtraction


def page(no: int, blocks: list[tuple[str, str]]) -> dict:
    """A stored page as ``_page_record`` writes it: heading blocks carry their ``#``."""
    return {"page_index": no, "markdown": "\n\n".join(t for _, t in blocks),
            "blocks": [{"block_id": f"p{no}_text_{i}", "page": no, "block_type": kind, "text": text}
                       for i, (kind, text) in enumerate(blocks, 1)]}


FOOTER = ("paragraph", "Oficina Municipal de Dades")


def pcap_pages() -> list[dict]:
    return [
        page(1, [("heading", "# 1. OBJECTE DEL CONTRACTE"), ("paragraph", "El contracte té per objecte…"), FOOTER]),
        page(2, [("heading", "# 2. PRESSUPOST"), ("heading", "### Total 1.000,00 € 2.000,00 €"),
                 ("paragraph", "1. La mesa de contractació estarà formada per:"), ("list_item", "- El president"), FOOTER]),
        page(3, [("heading", "## 2.1. Valor estimat"), ("paragraph", "3. SOLVÈNCIA"), ("paragraph", "a) punt"), FOOTER]),
        page(4, [("paragraph", "Annex sense estil"), ("paragraph", "Text de l'annex"), FOOTER]),
    ]


def test_collect_keeps_numbered_headings_and_drops_footers_tables_and_list_items():
    cands, report = collect(pcap_pages())
    texts = {c.text: c for c in cands}
    assert "Oficina Municipal de Dades" not in texts and "Total 1.000,00 € 2.000,00 €" not in texts
    assert texts["1. OBJECTE DEL CONTRACTE"].known and texts["2. PRESSUPOST"].known and texts["2.1. Valor estimat"].known
    assert texts["2.1. Valor estimat"].level_hint == 2
    # A numbered paragraph is a candidate for the judge, not a heading by itself.
    mesa = texts["1. La mesa de contractació estarà formada per:"]
    assert not mesa.known and mesa.repeats and mesa.following.startswith("- El president")
    assert texts["3. SOLVÈNCIA"].known is False and texts["3. SOLVÈNCIA"].numbered
    # A footer that is never a heading block nor numbered is no candidate at all.
    assert report["dropped"]["repeated"] == 0 and report["known"] == 3 and report["candidates"] == 6


def test_rules_decide_without_a_judge_and_the_tree_nests_by_numbering():
    structure, report = build_headings_structure(pcap_pages(), ["p1", "p2", "p3", "p4"])
    titles = [n["title"] for n in structure]
    assert titles == ["1. OBJECTE DEL CONTRACTE", "2. PRESSUPOST", "3. SOLVÈNCIA"]
    # The parent opens on page 2 before its first child on page 3: an intro node first, as flash does.
    assert [n["title"] for n in structure[1]["nodes"]] == ["2. PRESSUPOST (intro)", "2.1. Valor estimat"]
    assert (structure[1]["start_index"], structure[1]["end_index"]) == (2, 3)
    assert structure[2]["end_index"] == 4
    assert report["judge"] is None and report["asked"] == 3 and report["by_rule"] == 3
    decisions = {d["text"]: d["decision"] for d in report["decisions"]}
    assert decisions["1. La mesa de contractació estarà formada per:"] == "body"
    assert decisions["Annex sense estil"] == "body"


def test_index_pages_place_entries_in_the_body_and_layout_titles_fill_gaps():
    entries = [("heading", f"Cláusula {i}. Apartado {i} ......... {i + 1}") for i in range(1, 11)]
    pages = [page(1, entries)] + [page(i + 1, [("paragraph", f"Cláusula {i}. Apartado {i}."), ("paragraph", "Cuerpo")])
                                  for i in range(1, 11)]
    layout = [{"title": "Cláusula 4. Apartado 4.", "start_index": 5, "end_index": 5},
              {"title": "Disposición final", "start_index": 11, "end_index": 11}]
    cands, report = collect(pages, layout)
    assert report["index_pages"] == [1]
    by_text = {c.text: c for c in cands}
    # The body line of clause 4 is confirmed by the index entry and by the layout title.
    assert by_text["Cláusula 4. Apartado 4."].sources == {"liteparse", "toc", "layout"}
    assert by_text["Disposición final"].sources == {"layout"} and not by_text["Disposición final"].known
    assert all(c.known for c in cands if "toc" in c.sources) and sum("toc" in c.sources for c in cands) == 10


def test_an_index_page_written_with_leaders_and_emphasis_is_located_not_copied():
    """The informe of Open Data BCN: page 2 lists «3. ***TÍTULO ........ 5» lines; they place the body headings,
    they never become nodes themselves, and the titles carry no asterisks."""
    entries = [("heading", f"# {i}. ***APARTAT {i} DEL DOCUMENT***......................... {i + 2}") for i in range(1, 8)]
    pages = [page(1, [("heading", "# INFORME JUSTIFICATIU")]), page(2, entries)]
    pages += [page(i + 2, [("heading", f"# {i}. APARTAT {i} DEL DOCUMENT"), ("paragraph", "Cos del text")]) for i in range(1, 8)]
    cands, report = collect(pages)
    assert report["index_pages"] == [2] and report["dropped"]["index_page"] == 7
    assert all(c.page != 2 for c in cands) and all("*" not in c.text for c in cands)
    structure, _ = build_headings_structure(pages, ["p"] * 9)
    # The two-word title of page 1 is no section: the pages before the first one are the preface, as in flash.
    assert [n["title"] for n in structure][:3] == ["Preface", "1. APARTAT 1 DEL DOCUMENT", "2. APARTAT 2 DEL DOCUMENT"]
    assert [n["start_index"] for n in structure][1:4] == [3, 4, 5]
    # The same line outside an index page, with leaders, is an index entry: never a heading.
    stray, _ = collect([page(1, [("heading", "# 4. ALTRE APARTAT ............ 9"), ("paragraph", "Text")])])
    assert stray == []


def test_levels_hang_clauses_under_chapters_and_annexes_at_the_top():
    def cand(i, text, page_no):
        from pageindex.headings.candidates import marker_of
        marker, numbered, keyword = marker_of(text)
        return Candidate(index=i, page=page_no, order=i, text=text, marker=marker, numbered=numbered, keyword=keyword,
                         decision="heading")
    heads = [cand(0, "CAPÍTULO I. DISPOSICIONES", 1), cand(1, "Cláusula 1. Objeto", 1), cand(2, "1.1 Alcance", 2),
             cand(3, "Cláusula 2. Plazo", 3), cand(4, "CAPÍTULO II. EJECUCIÓN", 4), cand(5, "Cláusula 3. Pagos", 4),
             cand(6, "ANEXO I. Modelo", 6)]
    tree = build_structure(heads, 7)
    assert [n["title"] for n in tree] == ["CAPÍTULO I. DISPOSICIONES", "CAPÍTULO II. EJECUCIÓN", "ANEXO I. Modelo"]
    chapter = tree[0]
    assert [n["title"] for n in chapter["nodes"]] == ["Cláusula 1. Objeto", "Cláusula 2. Plazo"]
    assert chapter["nodes"][0]["nodes"][0]["title"] == "1.1 Alcance"
    assert (chapter["start_index"], chapter["end_index"]) == (1, 4) and tree[2]["end_index"] == 7


class FakeSession:
    def __init__(self, answers):
        self.bodies = []
        self._answers = answers

    def post(self, url, json, timeout, headers):
        self.bodies.append((url, json, headers))
        session = self

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"answers": session._answers(json), "usage": {"input_tokens": 1200, "output_tokens": 20}}
        return Response()


def test_typesafe_judge_asks_groups_of_siblings_and_reads_the_choice():
    pages = pcap_pages()
    cands, _ = collect(pages)
    ask = [c for c in cands if not c.known]

    def answers(body):
        assert body["model"] == "jev-latest"
        assert body["state"]["convention"].startswith("Un apartado")
        assert {line["index"] for line in body["state"]["outline"]} >= {c.index for c in ask}
        return {qid: {"type": "choice", "choice": "heading" if "SOLV" in json.dumps(body["state"]["lines"][i], ensure_ascii=False) else "body",
                      "probabilities": {"heading": 0.91, "body": 0.09}}
                for i, qid in enumerate(body["questions"])}

    session = FakeSession(answers)
    judge = TypesafeHeadingJudge(api_key="key", session=session)
    decided = judge.decide({"name": "pcap.pdf"}, cands, ask)
    by_text = {c.text: decided[c.index] for c in ask}
    assert by_text["3. SOLVÈNCIA"] == ("heading", 0.91)
    assert by_text["1. La mesa de contractació estarà formada per:"][0] == "body"
    assert session.bodies[0][2] == {"Authorization": "Bearer key"} and session.bodies[0][0].endswith("/v1/systemone")
    assert judge.usage["requests"] == 1 and judge.usage["input_tokens"] == 1200


def test_the_judge_adds_with_a_fair_margin_and_removes_only_when_sure():
    from pageindex.headings import _settle
    assert _settle("body", "heading", 0.7) == ("heading", "judge:0.70")
    assert _settle("body", "heading", 0.5)[0] == "body"
    assert _settle("heading", "body", 0.9) == ("body", "judge:0.90")
    assert _settle("heading", "body", 0.7)[0] == "heading"
    assert _settle("heading", "heading", 0.55) == ("heading", "judge:0.55")


def test_a_failed_judge_request_leaves_the_lines_to_the_rules():
    class Broken:
        name = "typesafe"

        def decide(self, document, candidates, ask):
            raise RuntimeError("down")
    structure, report = build_headings_structure(pcap_pages(), ["p1", "p2", "p3", "p4"], judge=Broken())
    assert [n["title"] for n in structure][:2] == ["1. OBJECTE DEL CONTRACTE", "2. PRESSUPOST"]
    assert report["judge_error"] == "down" and report["by_rule"] == 3


def test_client_indexes_in_headings_mode_with_the_flash_summaries(tmp_path, sample_pdf, monkeypatch):
    import pageindex.flash
    import pageindex.flash.api
    import pageindex.pdf_extraction
    import pageindex.utils

    pages = pcap_pages()[:2]
    for p in pages:
        p["markdown"] = p["markdown"] or "x"
    monkeypatch.setattr(pageindex.pdf_extraction, "extract_liteparse",
                        lambda path, conf: PDFExtraction(pages, {"parser": "liteparse", "layout": True, "ocr_pages": []}))
    seen = {}

    def fake_flash(pdf, **kwargs):
        seen.update(kwargs)
        return {"structure": [{"title": "2. PRESSUPOST", "start_index": 2, "end_index": 2}], "toc_source": "detected"}
    monkeypatch.setattr(pageindex.flash, "page_index_flash", fake_flash)
    summarized = []

    async def fake_summarize(structure, page_list, model, **kwargs):
        summarized.append((model, kwargs.get("max_words")))
        for node in structure:
            node["summary"] = f"resumen de {node['title']}"
    monkeypatch.setattr(pageindex.flash.api, "_summarize", fake_summarize)
    monkeypatch.setattr(pageindex.utils, "generate_doc_description", lambda structure, model=None: "PCAP de prueba")

    client = PageIndexClient(index={"mode": "local", "model": "openai/gpt-test", "storage_path": str(tmp_path / "store"),
                                    "pdf_parser": "liteparse"})
    result = client.submit_document(sample_pdf, mode="headings", doc_id="pcap")
    tree = client.get_tree("pcap", node_summary=True, include_text=False)["result"]
    assert [n["title"] for n in tree] == ["1. OBJECTE DEL CONTRACTE", "2. PRESSUPOST"]
    assert tree[0]["summary"] == "resumen de 1. OBJECTE DEL CONTRACTE" and tree[0]["node_id"] == "0000"
    assert summarized == [("openai/gpt-test", 150)]
    assert seen == {"summary": False, "optimize": False, "use_embedded_toc": True}
    doc = client.get_document("pcap")
    assert doc["description"] == "PCAP de prueba"
    features = client.get_tree("pcap", node_summary=False, include_text=False)["features"]
    assert features["pdf_extraction"]["structure_backend"] == "headings"
    assert features["pdf_extraction"]["headings"]["known"] == 2 and "decisions" not in features["pdf_extraction"]["headings"]


def test_headings_mode_needs_liteparse_and_a_known_judge(tmp_path, sample_pdf):
    client = PageIndexClient(storage_path=str(tmp_path / "store"))
    with pytest.raises(PageIndexAPIError, match="pdf_parser='liteparse'"):
        client.submit_document(sample_pdf, mode="headings")
    with pytest.raises(PageIndexAPIError, match="heading_judge"):
        PageIndexClient(index={"mode": "local", "model": "openai/x", "storage_path": str(tmp_path / "s2"),
                               "pdf_parser": "liteparse", "heading_judge": "oracle"})._api._heading_judge_instance()
    with pytest.raises(PageIndexAPIError, match="TYPESAFE_AI_API_KEY"):
        PageIndexClient(index={"mode": "local", "model": "openai/x", "storage_path": str(tmp_path / "s3"),
                               "pdf_parser": "liteparse", "heading_judge": "typesafe"})._api._heading_judge_instance()
