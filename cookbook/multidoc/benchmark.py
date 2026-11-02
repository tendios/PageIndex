"""Controlled, synthetic 10-document benchmark; no indexing/model calls unless --run.

The two arms share the exact stored trees/text/model. Gold answers are not sent
to the model. This measures query behavior, not OCR or index quality. Review
answers manually as well as the deliberately narrow deterministic checks.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import random
import re
import threading
import time
from pathlib import Path

from pageindex import MultiDocumentOptions, MultiDocumentSearch, PageIndexClient
from pageindex.multidoc import cited_ids
from pageindex.stores import FileDocumentStore

NAMES = ["Alfa", "Beta", "Gamma", "Delta", "Epsilon", "Zeta", "Eta", "Theta", "Iota", "Kappa"]


def fixture(path):
    store = FileDocumentStore(str(path))
    for i, name in enumerate(NAMES):
        doc_id = f"contract-{i + 1:02d}"
        pages = []
        for page in range(1, 31):
            text = (f"Proyecto {name}. Cláusula administrativa {page}. La documentación se presentará "
                    "por registro electrónico. Las comunicaciones se enviarán al responsable del contrato. "
                    "Las obligaciones de esta cláusula se aplican exclusivamente al proyecto indicado.")
            title = f"Gestión documental {page}"
            if page == 6:
                title, text = "Plazo de pago", f"Proyecto {name}. El plazo de pago es de {15 + i * 5} días naturales desde la aprobación de la factura."
            if page == 12:
                title, text = "Aval", f"Proyecto {name}. El aval exigido asciende a {10000 + i * 5000} euros."
            if page == 22:
                title = "Excepción al horario"
                text = f"Proyecto {name}. El servicio se presta de lunes a viernes de 08:00 a 16:00."
                if i == 0:
                    text += " Durante agosto, los viernes el servicio termina a las 13:00 y no se exige sustitución."
            if page == 28:
                title, text = "Prórroga", f"Proyecto {name}. La prórroga requiere acuerdo expreso de ambas partes; no se admite prórroga tácita."
            if i == 9 and page == 29:
                title, text = "Depósito singular", "Proyecto Kappa. Se exige un depósito de 7.350 euros antes del 18 de diciembre de 2026. Este depósito es adicional al aval."
            if i == 2 and page in (16, 27):
                title, text = ("Plazo de entrega", "Proyecto Gamma. La cláusula principal fija la entrega en 40 días.") if page == 16 else (
                    "Anexo de entrega", "Proyecto Gamma. El anexo fija la entrega en 55 días. No se establece una regla de prelación entre el anexo y la cláusula principal.")
            pages.append({"page_index": page, "markdown": text, "_title": title})
        # Full leaf summaries make the navigation stress explicit. The hierarchy
        # remains realistic enough to test a deep exception and an annex.
        leaves = [{"node_id": f"p{p['page_index']}", "title": p.pop("_title"),
                   "start_index": p["page_index"], "end_index": p["page_index"],
                   "summary": p["markdown"]} for p in pages]
        tree = [{"node_id": "root", "title": f"Condiciones del proyecto {name}", "summary": "Condiciones económicas, prestación, entrega y anexos.",
                 "start_index": 1, "end_index": 30, "nodes": [
                     {"node_id": "conditions", "title": "Condiciones y excepciones", "start_index": 1,
                      "end_index": 24, "summary": "Obligaciones y condiciones de ejecución.", "nodes": leaves[:24]},
                     {"node_id": "annexes", "title": "Anexos", "start_index": 25, "end_index": 30,
                      "summary": "Condiciones complementarias.", "nodes": leaves[24:]}]}]
        meta = {"id": doc_id, "name": f"Contrato_{name}.pdf", "description": f"Condiciones del proyecto {name}",
                "createdAt": "2026-11-02T00:00:00.000", "pageNum": 30, "status": "completed", "metadata": {"project": name}}
        store.save_document(doc_id, meta, tree, pages)
    return store


QUESTIONS = [
    {"id": "last_document", "question": "¿Qué depósito adicional al aval exige Kappa y antes de qué fecha? Cita la fuente.",
     "checks": ["7.350|7350", "18", "diciembre|12", "2026"], "sources": [["contract-10", 29]]},
    {"id": "deep_exception", "question": "¿Qué horario cumple Alfa los viernes de agosto y necesita sustitución? Cita literalmente la excepción.",
     "checks": ["13:00|13.00|13 horas", "no se exige sustitución|sin sustitución|no.*sustitución"], "sources": [["contract-01", 22]]},
    {"id": "compare_two", "question": "Compara los plazos de pago de Alfa y Beta, indicando desde cuándo se cuentan.",
     "checks": ["15", "20", "naturales", "aprobación"], "sources": [["contract-01", 6], ["contract-02", 6]]},
    {"id": "all_ten", "question": "Haz una tabla del plazo de pago de los diez proyectos. No omitas ninguno y cita la fuente de cada plazo.",
     "checks": NAMES + [str(15 + i * 5) for i in range(10)], "sources": [[f"contract-{i+1:02d}", 6] for i in range(10)]},
    {"id": "literal_quote", "question": "Transcribe literalmente la condición completa de prórroga de Theta, incluyendo si se admite tácitamente.",
     "checks": ["La prórroga requiere acuerdo expreso de ambas partes; no se admite prórroga tácita."], "sources": [["contract-08", 28]]},
    {"id": "multi_hop", "question": "¿Qué proyectos exigen un aval de al menos 30.000 euros y a la vez un plazo de pago de como máximo 40 días? Indica ambos valores y las fuentes.",
     "checks": ["Epsilon", "Zeta", "30.000|30000", "35.000|35000", "35", "40"],
     "sources": [["contract-05", 6], ["contract-05", 12], ["contract-06", 6], ["contract-06", 12]]},
    {"id": "contradiction", "question": "¿Cuál es el plazo de entrega de Gamma? Contrasta cláusula principal y anexo, y explica si puede decidirse cuál prevalece.",
     "checks": ["40", "55", "no.*prelación|no.*prevalece|no.*prevalencia|no.*prioridad|no.*determinar"], "sources": [["contract-03", 16], ["contract-03", 27]]},
    {"id": "absence", "question": "¿Qué certificación Platinum-2027 exigen estos contratos? Si no encuentras base documental, dilo sin inventarla.",
     "checks": ["no.*(?:encontr|identific|mencion|consta|evidencia|información|referencia)"], "sources": []},
]


def evaluate(answer, citations, item):
    checks = [bool(re.search(pattern, answer, re.I | re.S)) for pattern in item["checks"]]
    actual = {(c["doc_id"], c["page"]) for c in citations}
    expected = {tuple(s) for s in item["sources"]}
    return {"checks_passed": sum(checks), "checks_total": len(checks), "checks": checks,
            "expected_sources_found": len(actual & expected), "expected_sources_total": len(expected),
            "manual_review_required": True}


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    store = fixture(args.output / "store")
    ids = [f"contract-{i+1:02d}" for i in range(10)]
    corpus = [{"meta": store.get_meta(d), "tree": store.get_tree(d), "pages": store.get_pages(d)} for d in ids]
    manifest = {"dataset": "synthetic controlled contracts; prebuilt index, no OCR/indexing measured",
                "documents": 10, "pages": 300, "model": args.model,
                "sha256": hashlib.sha256(json.dumps(corpus, sort_keys=True).encode()).hexdigest(),
                "questions": QUESTIONS, "repeats": args.repeats, "seed": 7,
                "multidoc_code_sha256": hashlib.sha256(Path(__file__).parents[2].joinpath("pageindex/multidoc.py").read_bytes()).hexdigest(),
                "checks": "Narrow keyword/reference checks plus manual review; not a semantic quality score"}
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    if not args.run:
        print("Fixture and manifest prepared; pass --run to make paid API calls.")
        return
    if args.env_file:
        from dotenv import load_dotenv
        load_dotenv(args.env_file, override=False)
    tasks = [(arm, q, repeat) for repeat in range(args.repeats) for q in QUESTIONS
             if not args.questions or q["id"] in args.questions.split(",") for arm in args.arms.split(",")]
    random.Random(7).shuffle(tasks)
    log = args.output / "runs.jsonl"
    done = {(r["mode"], r["question_id"], r["repeat"]) for line in log.read_text().splitlines()
            if (r := json.loads(line))} if log.exists() else set()
    lock = threading.Lock()

    def one(task):
        arm, item, repeat = task
        if (arm, item["id"], repeat) in done:
            return
        client = PageIndexClient(index={"store": store}, chat=args.model)
        started = time.monotonic()
        row = {"mode": arm, "question_id": item["id"], "repeat": repeat}
        try:
            if arm == "multi_document":
                result = MultiDocumentSearch(client, options=MultiDocumentOptions(timeout_seconds=300)).query(
                    item["question"], doc_ids=ids, include_trace=True)
                answer = result["answer"]
                cited = set(cited_ids(answer))
                citations = [c for c in result["citations"] if c["id"] in cited]
                row.update(status=result["status"], usage=result["usage"], result=result)
            else:
                result = client.chat(item["question"], doc_id=ids, protocol="chat_completions", citations=True, max_turns=20)
                answer = result["choices"][0]["message"]["content"] or ""
                citations = client.get_citations(answer, doc_id=ids)
                row.update(status="complete" if result["choices"][0].get("finish_reason") == "stop" else "partial",
                           usage=result.get("usage"), result=dict(result))
            row.update(answer=answer, citations=citations, evaluation=evaluate(answer, citations, item))
        except Exception as exc:
            row.update(status="failed", error=f"{type(exc).__name__}: {exc}", usage=getattr(exc, "usage", None))
        row["seconds"] = round(time.monotonic() - started, 3)
        with lock:
            with log.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
            print(arm, item["id"], repeat, row["status"], row["seconds"], row.get("evaluation", {}), flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(one, tasks))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="anthropic/claude-haiku-5-5")
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--questions", default="")
    parser.add_argument("--arms", default="standard,multi_document")
    run(parser.parse_args())
