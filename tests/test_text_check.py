"""The text check of the LiteParse read (pageindex.text_check): what LiteParse loses found against PDFium's text
layer, put back where it was in every view of the page, the fallback re-reads and their choice, the headings that
never see recovered lines, and the pages that stay byte for byte as they were read."""
import copy
import json
from dataclasses import asdict

import pytest

from pageindex import text_check as tc
from pageindex.pdf_extraction import (_page_record, _swallowing_table, _text_format, extract_liteparse, locate_text,
                                      parser_config)

CONF = parser_config("liteparse")
DISABLED = {"name": "liteparse", "ocr_mode": "disabled"}

# ── a PDF where LiteParse loses a line: a hyphen in a font with Cambria Math's ascent and descent ────────────────

PARAGRAPH = [
    "El licitador debe presentar la oferta tecnica dentro del plazo establecido en el anuncio",
    "de licitacion, junto con la documentacion administrativa que exige el pliego de clausulas",
    "y con la declaracion responsable firmada por el representante legal de la empresa ofertante.",
    "La arquitectura del procesador debe ser x86-64 o equivalente con la misma familia de instrucciones",
    "porque el sistema operativo y las aplicaciones de gestion se ejecutan sobre esa plataforma",
    "sin necesidad de emulacion ni de adaptaciones posteriores por parte de la entidad contratante.",
    "Ademas la cabina de almacenamiento debe ofrecer alta disponibilidad con dos controladoras activas",
    "y protocolos estandar de acceso a bloques y ficheros para todos los servidores del centro.",
    "Las ofertas que no cumplan los requisitos minimos seran excluidas del procedimiento de adjudicacion",
    "y el organo de contratacion lo comunicara a los licitadores afectados en el plazo de diez dias."]
FIRST = ["El licitador debe presentar la oferta tecnica dentro del plazo establecido en el anuncio",
         "de licitacion, junto con la documentacion administrativa que exige el pliego de clausulas."]
SECOND = ["La arquitectura del procesador debe ser x86-64 o equivalente con la misma familia de instrucciones",
          "porque el sistema operativo y las aplicaciones de gestion se ejecutan sobre esa plataforma",
          "sin necesidad de emulacion ni de adaptaciones posteriores por parte de la entidad contratante."]
THIRD = ["Ademas la cabina de almacenamiento debe ofrecer alta disponibilidad con dos controladoras activas",
         "y protocolos estandar de acceso a bloques y ficheros para todos los servidores del centro."]


def _escape(text):
    return text.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")


def tall_glyph_pdf(pages):
    """A PDF of Helvetica lines, each page a list of (y, text); a «@» in a line is a hyphen set in a TrueType font
    whose descriptor has Cambria Math's ascent (3116) and descent (-2463): its box is 56 pt tall at 10 pt, as in the
    tribunal resolutions where LiteParse loses the lines around it."""
    n = len(pages)
    objects = [b"<< /Type /Catalog /Pages 2 0 R >>",
               ("<< /Type /Pages /Kids [%s] /Count %d >>" % (" ".join(f"{3 + i} 0 R" for i in range(n)), n)).encode()]
    fonts = 3 + 2 * n
    for i in range(n):
        objects.append((f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 {fonts} 0 R "
                        f"/F2 {fonts + 1} 0 R >> >> /Contents {3 + n + i} 0 R >>").encode())
    for lines in pages:
        parts = []
        for y, text in lines:
            if "@" in text:
                a, b = text.split("@")
                parts.append(f"BT /F1 10 Tf 72 {y:.1f} Td ({_escape(a)}) Tj /F2 10 Tf (-) Tj "
                             f"/F1 10 Tf ({_escape(b)}) Tj ET")
            else:
                parts.append(f"BT /F1 10 Tf 72 {y:.1f} Td ({_escape(text)}) Tj ET")
        stream = " ".join(parts).encode()
        objects.append(b"<< /Length %d >>\nstream\n%s\nendstream" % (len(stream), stream))
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    objects.append(f"<< /Type /Font /Subtype /TrueType /BaseFont /CambriaMath /FirstChar 45 /LastChar 45 "
                   f"/Widths [333] /FontDescriptor {fonts + 2} 0 R >>".encode())
    objects.append(b"<< /Type /FontDescriptor /FontName /CambriaMath /Flags 6 /FontBBox [-1474 -2463 2867 3117] "
                   b"/ItalicAngle 0 /Ascent 3116 /Descent -2463 /CapHeight 3116 /StemV 80 >>")
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % number + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    for offset in offsets:
        out += b"%010d 00000 n \n" % offset
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    return bytes(out)


def laid_out(paragraphs, pitch=14.4, gap=2.2, top=700.0):
    """Lines (y, text) of paragraphs set at `pitch`, `gap` pitches between paragraphs."""
    lines, y = [], top
    for paragraph in paragraphs:
        for text in paragraph:
            lines.append((y, text))
            y -= pitch
        y -= (gap - 1) * pitch
    return lines


def with_tall_hyphen(lines, k, word=3):
    """The k-th line with the tall hyphen after its word `word`."""
    lines = list(lines)
    y, text = lines[k]
    words = text.split(" ")
    words[word] += "@x"
    lines[k] = (y, " ".join(words))
    return lines


def squash(text):
    return " ".join((text or "").split())


def plain_read(data):
    """LiteParse's read as the fork reads, without the check: the page dicts."""
    from liteparse import LiteParse
    return asdict(LiteParse(ocr_enabled=False, output_format="markdown", keep_headers_footers=True, extract_blocks=True,
                            emit_word_boxes=True, extract_content_bounds=True, preserve_very_small_text=True,
                            quiet=True).parse(data))["pages"]


def read_with_check(tmp_path, data, name="tall.pdf"):
    path = tmp_path / name
    path.write_bytes(data)
    return extract_liteparse(str(path), DISABLED, heading_hints=False)


def real_reader():
    pytest.importorskip("liteparse")
    pytest.importorskip("pdf_inspector")
    pytest.importorskip("pypdfium2")


def lost_by_liteparse(pages, line):
    text = squash(pages[0]["text"])
    blocks = squash(" ".join(b.get("text") or "" for b in pages[0]["blocks"] or []))
    if squash(line) in text or squash(line) in blocks:
        pytest.skip("this LiteParse no longer loses the line under a tall glyph")


# ── real LiteParse, real PDFium ──────────────────────────────────────────────────────────────────────────────


def test_a_line_a_tall_glyph_swallows_goes_back_inside_its_paragraph_in_every_view(tmp_path, monkeypatch):
    real_reader()
    data = tall_glyph_pdf([with_tall_hyphen(laid_out([PARAGRAPH]), 1)])
    lost = "de licitacion, junto con-x la documentacion administrativa que exige el pliego de clausulas"
    lost_by_liteparse(plain_read(data), lost)

    result = read_with_check(tmp_path, data)

    entry = result.metadata["text_check"]["pages"][0]
    assert (entry["action"], entry["read"], entry["placed"]) == ("recovered", "liteparse", {"inside": 1})
    assert entry["missing"] == {"before": 14, "after": 0} and result.metadata["text_check"]["checked"] == 1
    page, record = result.liteparse["document"]["pages"][0], result.pages[0]
    around = f"{squash(PARAGRAPH[0])} {lost} {squash(PARAGRAPH[2])}"
    # LiteParse's blocks and markdown, the stored page and its citation markdown: the line between its neighbours
    assert around in squash(" ".join(b["text"] for b in page["blocks"]))
    assert around in squash(page["markdown"]) and around in squash(record["markdown"])
    assert around in squash(record["citation_markdown"]) and around in squash(result.liteparse["document"]["text"])
    # the spatial text: a row of its own between the rows of its neighbours
    rows = [squash(r) for r in page["text"].split("\n") if r.strip()]
    assert rows[:3] == [squash(PARAGRAPH[0]), lost, squash(PARAGRAPH[2])]
    # the text items: the line's own item, in its place, with PDFium's word boxes and no font metadata
    items = page["text_items"]
    k = next(i for i, t in enumerate(items) if t.get("recovered"))
    assert squash(items[k]["text"]) == lost and squash(items[k - 1]["text"]) == squash(PARAGRAPH[0])
    assert items[k]["font_name"] is None and items[k]["font_size"] is None and items[k]["confidence"] is None
    assert len(items[k]["words"]) == len(lost.split()) and 95 < items[k]["y"] < 105
    # a citation finds a quote across the recovered line and its neighbour
    hits = locate_text(result.pages, "en el anuncio de licitacion, junto")
    assert len(hits) == 1 and hits[0]["boxes_points"][-1]["y"] == pytest.approx(items[k]["y"], abs=1)
    assert record["extraction"]["text_check"] == {"action": "recovered", "read": "liteparse", "lines": 1}
    json.dumps(result.pages, allow_nan=False)
    # the page as the headings read it is the page the read gave, before the recovery
    monkeypatch.setattr(tc, "check_pages", lambda *a, **k: ({"checked": 0, "pages": []}, {}))
    unchecked = read_with_check(tmp_path, data, "unchecked.pdf").pages[0]
    as_read = tc.as_read(record)
    assert {**as_read, "extraction": unchecked["extraction"]} == unchecked


def test_a_paragraphs_lost_first_lines_open_the_block_after_them(tmp_path):
    real_reader()
    data = tall_glyph_pdf([with_tall_hyphen(laid_out([FIRST, SECOND, THIRD]), 3)])
    lost_by_liteparse(plain_read(data), SECOND[0])

    result = read_with_check(tmp_path, data)

    entry = result.metadata["text_check"]["pages"][0]
    assert entry["placed"] == {"prepend": 1} and entry["lost"] == {"lines": 2, "words": 31}
    blocks = [b["text"] for b in result.liteparse["document"]["pages"][0]["blocks"]]
    assert squash(blocks[0]) == squash(" ".join(FIRST)) and squash(blocks[2]) == squash(" ".join(THIRD))
    assert squash(blocks[1]).startswith(squash(SECOND[0])) and squash(blocks[1]).endswith(squash(SECOND[2]))


def test_a_paragraph_split_at_its_lost_line_gets_it_back_at_the_end_of_its_first_half(tmp_path):
    real_reader()
    lines = with_tall_hyphen(laid_out([FIRST + SECOND + THIRD]), 1)
    data = tall_glyph_pdf([lines])
    lost_by_liteparse(plain_read(data), "de licitacion, junto con-x la documentacion")

    result = read_with_check(tmp_path, data)

    assert result.metadata["text_check"]["pages"][0]["placed"] == {"append": 1}
    first = result.liteparse["document"]["pages"][0]["blocks"][0]
    assert squash(first["text"]).endswith("de licitacion, junto con-x la documentacion administrativa que exige el "
                                          "pliego de clausulas.")
    assert first["recovered"] == [[len(squash(FIRST[0])), len(first["text"]), ""]]


def test_clean_pages_stay_byte_for_byte_as_read(tmp_path, monkeypatch):
    real_reader()
    from conftest import build_pdf
    clean = [[(72, 720, 11, "1. Objeto del contrato"), (72, 700, 11, "El contrato tiene por objeto el suministro "
                                                                         "de material de oficina para las sedes.")],
             [(72, 720, 11, "2. Precio"), (72, 700, 11, "El precio es de mil euros por cada lote adjudicado.")]]
    tall = tall_glyph_pdf([with_tall_hyphen(laid_out([PARAGRAPH]), 1), laid_out([FIRST, THIRD])])
    for name, data in (("clean.pdf", build_pdf(clean)), ("tall.pdf", tall)):
        path = tmp_path / name
        path.write_bytes(data)
        checked = extract_liteparse(str(path), DISABLED)
        with monkeypatch.context() as m:
            m.setattr(tc, "check_pages", lambda *a, **k: ({"checked": 0, "pages": []}, {}))
            unchecked = extract_liteparse(str(path), DISABLED)
        changed = {e["page"] for e in checked.metadata["text_check"]["pages"]}
        assert changed == (set() if name == "clean.pdf" else {1})
        for before, after in zip(unchecked.pages, checked.pages):
            if before["page_index"] not in changed:
                assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True)
        for before, after in zip(unchecked.liteparse["document"]["pages"], checked.liteparse["document"]["pages"]):
            if before["page_num"] not in changed:
                assert json.dumps(after, sort_keys=True) == json.dumps(before, sort_keys=True)
        if not changed:
            assert checked.liteparse["document"] == unchecked.liteparse["document"]


def test_text_a_clip_path_hides_is_not_lost(tmp_path):
    real_reader()
    from conftest import build_pdf
    hidden = "Este texto queda fuera del recorte y no se ve nunca en la pagina."
    stream = (b"BT /F1 11 Tf 72 720 Td (El contrato tiene por objeto el suministro de material de oficina.) Tj ET "
              b"q 72 600 100 10 re W n BT /F1 11 Tf 72 500 Td (" + hidden.encode() + b") Tj ET Q")
    path = tmp_path / "clipped.pdf"
    path.write_bytes(build_pdf([stream]))

    result = extract_liteparse(str(path), DISABLED)

    assert result.metadata["text_check"]["pages"][0]["action"] == "none"
    assert result.metadata["text_check"]["pages"][0]["clipped_words"] == 14
    assert hidden not in result.texts[0] and "text_check" not in result.pages[0]["extraction"]


def test_an_unreadable_pdf_for_pdfium_leaves_the_read_as_it_is(tmp_path, monkeypatch):
    pages = {1: {"page_num": 1, "width": 200, "height": 300, "text": "x", "blocks": []}}
    report, changed = tc.check_pages(b"not a pdf", pages, [1], lambda *a: pytest.fail("no reread"), ocr=True,
                                     text_format=lambda p: "blocks", swallowing=lambda *a: False)
    assert changed == {} and report["checked"] == 0 and "error" in report


# ── synthetic pages: PDFium's lines and LiteParse's read made by hand ────────────────────────────────────────


def ref_lines(texts, top=100.0, pitch=14.4, gaps=()):
    """PDFium lines of 10 pt text: a box per character, one baseline each; `gaps` the line indexes a paragraph gap
    comes before."""
    lines, y = [], top
    for k, text in enumerate(texts):
        if k in gaps:
            y += pitch
        boxes = [None if c.isspace() else (72 + 5 * i, y, 77 + 5 * i, y + 8) for i, c in enumerate(text)]
        lines.append(tc._Line(text, bbox=(72, y, 72 + 5 * len(text), y + 8), baseline=y + 8, size=10, boxes=boxes))
        y += pitch
    return lines


def lp_page(lines, paragraphs, lost):
    """LiteParse's page of these lines: `paragraphs` lists each block's line indexes, `lost` the lines it dropped."""
    blocks, rows, items = [], [], []
    for group in paragraphs:
        kept = [lines[i] for i in group if i not in lost]
        if not kept:
            continue
        x0, y0 = kept[0].bbox[0], kept[0].bbox[1]
        blocks.append({"kind": "paragraph", "text": tc._join([ln.text for ln in kept]),
                       "bbox": {"x": x0, "y": y0, "width": 400, "height": kept[-1].bbox[3] - y0}})
    for i, ln in enumerate(lines):
        rows.append("" if i in lost else ln.text)
        if i not in lost:
            items.append({"text": ln.text, "x": ln.bbox[0], "y": ln.bbox[1], "width": 400, "height": 10,
                          "font_name": "Helvetica", "font_size": 10.0, "rotation": 0.0, "words": []})
    return {"page_num": 1, "width": 612, "height": 792, "text": "\n".join(rows),
            "markdown": "\n\n".join(b["text"] for b in blocks), "blocks": blocks, "text_items": items}


def recovered(lines, page):
    lost = tc._lost(lines, tc._Read(page))
    pieces = tc._pieces(lines, lost, 0.0)
    return tc._recover(page, lines, pieces), pieces


TEXTS = ["El licitador debe presentar la oferta tecnica dentro del plazo",
         "establecido en el anuncio de licitacion junto con la documentacion",
         "administrativa que exige el pliego de clausulas particulares del",
         "contrato y la declaracion responsable firmada por su representante.",
         "La arquitectura del procesador debe ser equivalente a la pedida",
         "porque las aplicaciones de gestion se ejecutan sobre esa plataforma."]


def test_a_lost_line_goes_inside_the_paragraph_that_holds_both_its_neighbours():
    lines = ref_lines(TEXTS, gaps=(4,))
    page = lp_page(lines, [[0, 1, 2, 3], [4, 5]], lost={2})

    new, pieces = recovered(lines, page)

    assert [p.place for p in pieces] == ["inside"]
    assert new["blocks"][0]["text"] == tc._join(TEXTS[:4])
    assert new["blocks"][0]["recovered"] == [[len(tc._join(TEXTS[:2])), len(tc._join(TEXTS[:3])) + 1, " "]]
    assert new["blocks"][1] == page["blocks"][1]
    assert new["markdown"] == tc._join(TEXTS[:4]) + "\n\n" + tc._join(TEXTS[4:])
    assert new["text"].split("\n") == TEXTS
    assert [t["text"] for t in new["text_items"]] == TEXTS and new["text_items"][2]["recovered"] is True
    assert tc.restore(new["blocks"][0]["text"], new["blocks"][0]["recovered"]) == page["blocks"][0]["text"]
    assert tc.restore(new["markdown"], new["recovered"]["markdown"]) == page["markdown"]
    assert tc.restore(new["text"], new["recovered"]["text"]) == page["text"]


def test_a_lost_paragraph_between_two_blocks_is_a_block_of_its_own():
    texts = TEXTS[:2] + ["Las ofertas que no cumplan los requisitos minimos seran excluidas."] + TEXTS[4:]
    lines = ref_lines(texts, gaps=(2, 3))
    page = lp_page(lines, [[0, 1], [2], [3, 4]], lost={2})

    new, pieces = recovered(lines, page)

    assert [p.place for p in pieces] == ["new"]
    assert [b["text"] for b in new["blocks"]] == [tc._join(texts[:2]), texts[2], tc._join(texts[3:])]
    assert new["blocks"][1]["recovered"] == [[0, len(texts[2]), ""]] and new["blocks"][1]["kind"] == "paragraph"
    assert new["markdown"].split("\n\n") == [b["text"] for b in new["blocks"]]
    # the blank row the read left for it holds it, the paragraph gaps stay
    assert new["text"].split("\n") == texts[:2] + [texts[2]] + texts[3:]


def test_a_lost_line_continues_the_word_a_line_end_hyphen_broke():
    texts = ["El licitador debe presentar la oferta tecnica dentro del pla-",
             "zo establecido en el anuncio de licitacion junto con la docu-",
             "mentacion administrativa que exige el pliego de clausulas."]
    lines = ref_lines(texts)
    page = lp_page(lines, [[0, 1, 2]], lost={1})
    assert page["blocks"][0]["text"].endswith("del plamentacion administrativa que exige el pliego de clausulas.")

    new, _ = recovered(lines, page)

    assert new["blocks"][0]["text"] == ("El licitador debe presentar la oferta tecnica dentro del plazo establecido "
                                        "en el anuncio de licitacion junto con la documentacion administrativa que "
                                        "exige el pliego de clausulas.")


def test_a_run_whose_anchors_are_not_side_by_side_is_not_placed():
    lines = ref_lines(TEXTS)
    page = lp_page(lines, [[0, 1, 2, 3, 4, 5]], lost={2})
    # the read has something else where the line was: both anchors are there, not side by side
    page["blocks"][0]["text"] = page["blocks"][0]["text"].replace("documentacion contrato",
                                                                  "documentacion y otra cosa contrato")

    new, pieces = recovered(lines, page)

    assert new is None and [p.reason for p in pieces] == ["apart"]


def test_a_line_whose_words_recur_elsewhere_counts_as_lost_where_it_is_missing():
    texts = ["Primero, el Tribunal Superior de Justicia de Andalucia dicto sentencia firme",
             "en el recurso que interpuso la empresa contra la resolucion de exclusion.",
             "Despues el Tribunal Superior de Justicia de Andalucia estimo el recurso",
             "y anulo la resolucion administrativa con retroaccion de las actuaciones."]
    lines = ref_lines(texts)
    page = lp_page(lines, [[0, 1, 2, 3]], lost={2})

    new, pieces = recovered(lines, page)

    assert [p.parts[0].whole for p in pieces] == [True] and new["blocks"][0]["text"] == tc._join(texts)


def test_a_line_the_read_kept_in_part_loses_only_its_missing_words():
    texts = ["Primero, el licitador presento su oferta ante la mesa de contratacion",
             "del Ayuntamiento Contencioso-Administrativo de la ciudad de Sevilla en plazo",
             "y el organo de contratacion la admitio despues de revisar toda la documentacion."]
    lines = ref_lines(texts)
    page = lp_page(lines, [[0, 1, 2]], lost=set())
    page["blocks"][0]["text"] = page["blocks"][0]["text"].replace("Contencioso-Administrativo ", "")
    page["text"] = page["text"].replace("Contencioso-Administrativo ", "")

    assert tc._lost(lines, tc._Read(page)) == []  # two words: no run, and the line is no lost line


# ── the fallback: the page alone read again, and the read kept ───────────────────────────────────────────────


def table_page(rows, prose):
    cells = [[{"text": c, "bbox": None} for c in row.split(" | ")] for row in rows]
    return {"page_num": 1, "width": 612, "height": 792, "text": "\n".join([prose] + rows),
            "markdown": prose + "\n\n" + "\n".join(rows),
            "blocks": [{"kind": "paragraph", "text": prose, "bbox": {"x": 72, "y": 60, "width": 400, "height": 30}},
                       {"kind": "table", "header": cells[0], "rows": cells[1:],
                        "bbox": {"x": 72, "y": 100, "width": 400, "height": 200}}],
            "text_items": [{"text": prose, "x": 72, "y": 60, "width": 400, "height": 10, "rotation": 0.0}]}


PROSE = "Presupuesto de ejecucion material de las obras de urbanizacion del sector norte"
ROWS = ["Partida | Unidades | Descripcion", "obr121 | 800,000 | tuberia de polietileno de doble pared roja",
        "obr159 | 8,000 | arqueta prefabricada de hormigon tipo A",
        "obr049 | 13,000 | arqueta prefabricada de hormigon tipo B"]


def pdf_of(lines):
    from conftest import build_pdf
    return build_pdf([[(72, 720 - 20 * k, 10, line) for k, line in enumerate(lines)]])


def run_check(lines, page, reads, ocr=True):
    """check_pages on a real PDF of `lines` (PDFium's side) and a read made by hand (LiteParse's); `reads` the pages
    each reread gives."""
    pytest.importorskip("pypdfium2")
    asked = []

    def reread(number, how):
        asked.append(how)
        return copy.deepcopy(reads[how] if how != "clipped" else page)

    pages = {1: copy.deepcopy(page)}
    report, changed = tc.check_pages(pdf_of(lines), pages, [1], reread, ocr=ocr,
                                     text_format=lambda p: _text_format(p, {"text_format": "auto"}),
                                     swallowing=_swallowing_table)
    return report["pages"][0], pages[1], asked


def test_lost_table_cells_take_the_text_metadata_read_when_its_structure_is_no_worse():
    lines = [PROSE] + [r.replace(" | ", " ") for r in ROWS]
    first = table_page([ROWS[0], ROWS[2], ROWS[3]], PROSE)
    better = table_page(ROWS, PROSE)

    entry, kept, asked = run_check(lines, first, {"text_metadata": better})

    assert asked == ["clipped", "text_metadata"]  # no OCR: nothing was left lost
    assert (entry["action"], entry["read"], entry["unplaced"]) == ("reread", "text_metadata", {"table": 1})
    assert entry["missing"] == {"before": 10, "after": 0} and kept["blocks"] == better["blocks"]


def test_a_read_that_moves_the_prose_into_a_table_loses_to_the_ocr_read():
    lines = [PROSE] + [r.replace(" | ", " ") for r in ROWS]
    first = table_page([ROWS[0], ROWS[2], ROWS[3]], PROSE)
    into_table = table_page([PROSE.replace(" ", " | ", 1)] + ROWS, "")
    into_table["blocks"] = into_table["blocks"][1:]
    good = table_page(ROWS, PROSE)

    entry, kept, asked = run_check(lines, first, {"text_metadata": into_table, "ocr": good})

    assert asked == ["clipped", "text_metadata", "ocr"]
    assert entry["tried"]["text_metadata"] == {"missing": 0, "worse": "prose_in_tables"}
    assert (entry["action"], entry["read"]) == ("reread", "ocr") and kept["blocks"] == good["blocks"]


def test_without_a_better_read_the_page_stays_as_read_and_says_so():
    lines = [PROSE] + [r.replace(" | ", " ") for r in ROWS]
    first = table_page([ROWS[0], ROWS[2], ROWS[3]], PROSE)

    entry, kept, asked = run_check(lines, first, {"text_metadata": first}, ocr=False)

    assert asked == ["clipped", "text_metadata"]  # OCR not allowed
    assert (entry["action"], entry["read"]) == ("unrecovered", "liteparse") and kept == first
    assert entry["missing"] == {"before": 10, "after": 10}


def structure(*blocks, text=""):
    return {"page_num": 1, "width": 612, "height": 792, "text": text, "blocks": list(blocks)}


def paragraph(text):
    return {"kind": "paragraph", "text": text, "bbox": {"x": 72, "y": 100, "width": 400, "height": 20}}


def table(*rows, height=200):
    cells = [[{"text": c, "bbox": None} for c in row] for row in rows]
    return {"kind": "table", "header": cells[0], "rows": cells[1:],
            "bbox": {"x": 0, "y": 0, "width": 612, "height": height}}


@pytest.mark.parametrize("name,candidate,expected", [
    ("same", structure(paragraph("uno dos tres cuatro"), table(["a", "b"], ["c", "d"])), None),
    ("a real table grows", structure(paragraph("uno dos tres cuatro"), table(["a", "b"], ["c", "d"], ["e", "f"],
                                                                             ["g", "h"], ["i", "j"])), None),
    ("prose into a table", structure(table(["uno dos", "tres cuatro"], ["c", "d"])), "prose_in_tables"),
    ("one table swallows the page", structure(paragraph("uno dos tres cuatro"),
                                              table(["x " * 700, "y"], height=700)), "one_table"),
    ("fragmented", structure(*(paragraph(w) for w in "uno dos tres cuatro cinco seis siete".split())), "fragmented"),
])
def test_what_makes_a_reads_structure_worse(name, candidate, expected):
    first = structure(paragraph("uno dos tres cuatro"), table(["a", "b"], ["c", "d"]))
    assert tc.worse_structure(candidate, first, lambda p: "blocks", _swallowing_table) == expected


def test_a_read_rendered_as_spatial_text_where_the_first_was_blocks_is_worse():
    first = structure(paragraph("uno dos tres cuatro"))
    spatial = structure(paragraph("uno dos tres cuatro"))
    formats = {id(first): "blocks", id(spatial): "spatial"}
    assert tc.worse_structure(spatial, first, lambda p: formats[id(p)], _swallowing_table) == "spatial"


# ── what the index stores, and the headings ──────────────────────────────────────────────────────────────────


def test_the_stored_page_marks_its_recovered_spans_after_a_headings_marks_and_in_a_spatial_page():
    heading = {"kind": "heading", "level": 2, "text": "Clausula primera recuperada", "recovered": [[8, 18, " "]],
               "bbox": {"x": 72, "y": 60, "width": 300, "height": 12}}
    page = {"page_num": 1, "width": 612, "height": 792, "text": "a\nrow", "blocks": [heading], "text_items": []}
    entry = _page_record(page, CONF, False)["blocks"][0]
    assert entry["text"] == "## Clausula primera recuperada"
    assert entry["text"][entry["recovered"][0][0]:entry["recovered"][0][1]] == heading["text"][8:18]

    spatial = {"page_num": 1, "width": 612, "height": 792, "text": "\n\nuno\nrecuperada\ndos", "blocks": [],
               "text_items": [], "recovered": {"text": [[6, 17, ""]], "markdown": []}}
    record = _page_record(spatial, CONF, False)
    s, e, was = record["blocks"][0]["recovered"][0]
    assert record["markdown"][s:e] == "recuperada\n" and was == ""
    assert tc.as_read(record)["markdown"] == "```text\nuno\ndos\n```"


def test_as_read_gives_the_page_liteparse_read():
    lines = ref_lines(TEXTS, gaps=(4,))
    page = lp_page(lines, [[0, 1, 2, 3], [4, 5]], lost={2})
    new, _ = recovered(lines, page)
    assert new["blocks"][0]["text"] != page["blocks"][0]["text"]
    assert tc.as_read(_page_record(new, CONF, False)) == _page_record(page, CONF, False)
    plain = _page_record(page, CONF, False)
    assert tc.as_read(plain) is plain


def test_recovered_lines_are_never_heading_candidates_and_never_move_one():
    from pageindex.headings import build_headings_structure

    def record(number, blocks):
        return _page_record({"page_num": number, "width": 612, "height": 792, "text": "", "text_items": [],
                             "blocks": blocks}, CONF, False)

    read = [record(1, [paragraph("1. Objeto del contrato"), paragraph("El contrato tiene por objeto el suministro.")]),
            record(2, [paragraph("2. Precio"), paragraph("El precio es de mil euros por cada lote.")])]
    put_back = copy.deepcopy(read)
    # a lost line that reads like a numbered title, as a block of its own and inside a paragraph
    lost = "3. Plazo de ejecucion del contrato y sus prorrogas"
    put_back[0]["blocks"].insert(1, {**put_back[0]["blocks"][0], "block_id": "p1_text_new", "text": lost,
                                     "recovered": [[0, len(lost), ""]]})
    text = put_back[1]["blocks"][1]["text"]
    put_back[1]["blocks"][1]["text"] = text + " 4. Garantias del contrato."
    put_back[1]["blocks"][1]["recovered"] = [[len(text), len(text) + 27, ""]]
    for p in put_back:
        p["markdown"] = "\n\n".join(b["text"] for b in p["blocks"])

    def titles(pages):
        structure, report = build_headings_structure(pages, [p["markdown"] for p in pages])
        return json.dumps(structure), [d["text"] for d in report["decisions"]]

    assert titles(put_back) == titles(read)
    assert not any("Plazo" in t or "Garantias" in t for t in titles(put_back)[1])


def test_the_document_text_is_joined_again_as_it_was_joined():
    before = [{"markdown": "uno", "text": "a"}, {"markdown": "dos", "text": "b"}]
    after = [{"markdown": "uno", "text": "a"}, {"markdown": "dos y tres", "text": "b c"}]
    assert tc.document_text("uno\n\n-----\n\ndos", before, after) == "uno\n\n-----\n\ndos y tres"
    assert tc.document_text("a\n\nb", before, after) == "a\n\nb c"
    assert tc.document_text("[uno] [dos]", before, after) == "[uno] [dos y tres]"


def test_the_check_reads_no_ocr_page_and_reports_how_many_it_checked(tmp_path, monkeypatch):
    from test_pdf_extraction import Result, fake_parser, page

    seen = []
    monkeypatch.setattr(tc, "check_pages", lambda data, pages, native, *a, **k: (
        seen.append(native) or ({"checked": len(native), "pages": []}, {})))
    path = tmp_path / "mixed.pdf"
    path.write_bytes(b"fixture")
    fake_parser(monkeypatch, [Result([page(1), page(2, ""), page(3)]), Result([page(2, "ocr")])])
    result = extract_liteparse(str(path), "liteparse")
    assert seen == [[1, 3]] and result.metadata["text_check"] == {"checked": 2, "pages": []}
