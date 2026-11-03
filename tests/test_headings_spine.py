"""Mode headings, second reading: the markers a document numbers its sections with, titles that run into a block,
the numbering spine, clause lists, inventory tables, and the page a giant table swallows."""
from __future__ import annotations

import pytest

from pageindex.headings import build_headings_structure
from pageindex.headings.numbering import follows, ordinal_value, parse
from pageindex.headings.structure import _split_inline, collect
from pageindex.pdf_extraction import _page_record, parser_config


@pytest.mark.parametrize("text, family, path", [
    ("Vint-i-setena. Responsabilitat", "ordinal", (27,)),
    ("Tretzena. Determinació de la millor oferta", "ordinal", (13,)),
    ("Vigésima primera. Penalidades", "ordinal", (21,)),
    ("DECIMOTERCERA.- Pagos", "ordinal", (13,)),
    ("B1. Determinació del preu", "letter", (2, 1)),
    ("B.1 ADMINISTRACIÓN CONTRATANTE", "letter", (2, 1)),
    ("3.1.A.-Condiciones específicas", "decimal", (3, 1, 1)),
    ("4.2.3.1.5 Requisitos no funcionales", "decimal", (4, 2, 3, 1, 5)),
    ("Clàusula 11. Mesa de contractació", "clause", (11,)),
    ("Sección primera. Del procedimiento", "chapter", (1,)),
    ("CAPÍTULO II. EJECUCIÓN", "chapter", (2,)),
    ("ANEXO I. Modelo", "annex", (1,)),
    ("2) Declaració responsable", "paren", (2,)),
])
def test_markers(text, family, path):
    m = parse(text)
    assert m is not None and (m.family, m.path) == (family, path)


@pytest.mark.parametrize("text", ["1.000,00 € total", "2024 se aprobó", "9 años de experiencia", "a) punt",
                                  "Anexo I de este Pliego para cada escuela", "14 GARANTÍAS"])
def test_not_markers(text):
    assert parse(text) is None  # «14 GARANTÍAS» only counts as a summary row (bare=True)


def test_bare_numbers_only_where_allowed_and_ordinals():
    assert parse("14 GARANTÍAS", bare=True).path == (14,)
    assert ordinal_value("Trenta-sisena") == 36 and ordinal_value("decimocuarta") == 14
    assert follows((2, 3), (2, 4)) and follows((2, 3), (2, 3, 1)) and follows((2, 3, 1), (3,))
    assert not follows((2, 3), (5,))


def test_titles_that_run_into_a_block_are_cut_but_references_are_not():
    assert _split_inline("IV. DISPOSICIONS Vint-i-sisena. Abonaments a l'empresa 26.1 L'import dels serveis") == [
        "IV. DISPOSICIONS", "Vint-i-sisena. Abonaments a l'empresa", "26.1 L'import dels serveis"]
    assert _split_inline("ingredientes de cada plato. 3.1.4. El adjudicatario elaborará") == [
        "ingredientes de cada plato.", "3.1.4. El adjudicatario elaborará"]
    assert _split_inline("según el artículo 22.1 La ley dice") == ["según el artículo 22.1 La ley dice"]


def page(no, blocks):
    return {"page_index": no, "markdown": "\n\n".join(t for _, t in blocks),
            "blocks": [{"block_id": f"p{no}_text_{i}", "page": no, "block_type": kind, "text": text}
                       for i, (kind, text) in enumerate(blocks, 1)]}


def test_spine_keeps_the_clauses_through_a_list_and_reads_a_clause_paragraph_list_as_content():
    pages = [
        page(1, [("heading", "## Clàusula 1. Objecte del contracte"),
                 ("paragraph", "1. Aquest contracte té per objecte el subministrament de materials per a les escoles."),
                 ("paragraph", "2. No s'estableix limitació als licitadors en quant a un número màxim de lots.")]),
        page(2, [("heading", "## Clàusula 2. Naturalesa jurídica"), ("paragraph", "Text del contracte.")]),
        page(3, [("heading", "## Clàusula 3. Valor estimat"), ("paragraph", "Text.")]),
    ]
    structure, report = build_headings_structure(pages, ["a", "b", "c"])
    assert [n["title"] for n in structure] == ["Clàusula 1. Objecte del contracte", "Clàusula 2. Naturalesa jurídica",
                                               "Clàusula 3. Valor estimat"]
    decisions = {d["text"][:12]: d["decision"] for d in report["decisions"]}
    assert decisions["1. Aquest co"] == "body" and decisions["2. No s'esta"] == "body"


def test_letters_in_a_lettered_sequence_are_not_roman_chapters_and_inventory_tables_are_skipped():
    pages = [page(1, [("heading", "# QUADRE DE CARACTERÍSTIQUES"), ("paragraph", "A. Objecte"),
                      ("paragraph", "B. Dades econòmiques"), ("paragraph", "C. Existència de crèdit"),
                      ("paragraph", "D. Termini de durada"),
                      ("table", "| | |\n| --- | --- |\n| 1 COCINA REPAGAS | x |\n| 1 SECADORA DOMUS | y |\n"
                                "| 2 BATIDORAS DCOOK | z |\n| 1 FREIDORA FAGOR | w |")])]
    cands, _, _ = collect(pages)
    by_text = {c.text: c for c in cands}
    assert by_text["C. Existència de crèdit"].mark.family == "letter"  # «C» after «B», not roman 100
    assert by_text["D. Termini de durada"].seq == "continue" and by_text["D. Termini de durada"].decision == "heading"
    assert not any("COCINA" in c.text for c in cands)


def test_a_page_a_giant_table_swallows_is_read_as_spatial_text():
    cell = {"text": "3.5 S'han complert tots els tràmits reglamentaris " * 40}
    raw = {"page_num": 4, "width": 600, "height": 800, "text": "3.5 S'han complert tots els tràmits", "text_items": [],
           "blocks": [{"kind": "table", "bbox": {"x": 10, "y": 10, "width": 580, "height": 760},
                       "rows": [[{"text": "DOCUMENTS"}], [cell]]}]}
    record = _page_record(raw, parser_config("liteparse"), False)
    assert record["extraction"]["text_format"] == "spatial" and record["blocks"][0]["block_type"] == "page"
