"""Library view benchmark: the cookbook's NVIDIA 10-K questions, each asked
with doc_id = its filings, with the library view off (one structure call
per document), on, and on with the topic index.

Per question it records structure calls (get_document_structure +
get_library_structure), pages read, tool calls, latency and tokens; the
answers are saved too, for the notebook's judge.

    pip install -U pageindex
    export OPENROUTER_API_KEY=...
    python library-view-benchmark.py --questions 63 --arms off,view,topics

It reuses the notebook's files: nvidia-10k/ (the filings, downloaded and
checked here if absent) and the pageindex-library/ store (indexed here if
absent). Results land in nvidia-10k/library-view-<arm>.jsonl; a rerun only
asks what is missing.
"""
import argparse
import hashlib
import json
import pathlib
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from pageindex import PageIndexClient

MODEL = "openrouter/openai/gpt-6-sol"           # as the notebook
INDEX_MODEL = "openrouter/openai/gpt-6-luna"
REASONING_EFFORT = "medium"
# OpenRouter USD per 1M tokens, as the notebook (checked 2026-10-16).
PRICE = {"input": 2.50, "cached": 0.20, "output": 10.00}

FILINGS = {
    2022: ("ca04d49f-caab-436f-9e6f-e8493c8e8515", "61ce01bbcd3886fa512dce61613a242a95c081d60b2f8bc0583b2f89ccad40eb"),
    2023: ("4e9abe7b-fdc7-4cd2-8487-dc3a99f30e98", "89981bbfcd91e20498c1060d7efb39022ac8f85e639f2841091695885aa9f8a8"),
    2024: ("1cbe8fe7-e08a-46e3-8dcc-b429fc06c1a4", "536f66d7f1c3413abbf643e0a02bd0aab65639116fe630225f3f93529244658b"),
    2025: ("177440d5-3b32-4185-8cc8-95500a9dc783", "b67bd67a64488a54886de001c788bc5059a965fc6ef5b4d8b71624951e13df8e"),
    2026: ("e361e58a-7483-44f5-bc62-a9080ae6ec72", "a056b59e4170c0123217a8e861a39af4a4a48d2df0d27b641017a618a280a396"),
}
HERE = pathlib.Path(__file__).parent
DATA = HERE / "nvidia-10k"
STORE = HERE / "pageindex-library"
STRUCTURE_TOOLS = {"get_document_structure", "get_library_structure"}


def filings():
    DATA.mkdir(exist_ok=True)
    for fy, (pdf_id, sha256) in FILINGS.items():
        path = DATA / f"NVIDIA_{fy}_10K.pdf"
        if not path.exists():
            response = requests.get(
                f"https://d18rn0p25nwr6d.cloudfront.net/CIK-0001045810/{pdf_id}.pdf", timeout=120)
            response.raise_for_status()
            path.write_bytes(response.content)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == sha256, path.name
        yield path


def make_client(arm):
    client = PageIndexClient(index_model=INDEX_MODEL, chat_model=MODEL,
                             storage_path=str(STORE), summary_max_words=70)
    client.library_view = arm != "off"
    client.library_topics = arm == "topics"
    return client


def page_count(spec):
    total = 0
    for part in str(spec).split(","):
        low, _, high = part.strip().partition("-")
        total += int(high or low) - int(low) + 1 if low.isdigit() else 0
    return total


def measure(client, question, doc_ids):
    start = time.perf_counter()
    response = client.chat_completions(question["question"], doc_id=doc_ids,
                                       reasoning_effort=REASONING_EFFORT,
                                       max_turns=20)
    latency = time.perf_counter() - start
    calls = [call["function"] for message in response["messages"]
             for call in message.get("tool_calls") or []]
    usage = response["usage"]
    return {
        "id": question["id"], "filings": len(question["filings"]),
        "structure_calls": sum(c["name"] in STRUCTURE_TOOLS for c in calls),
        "library_calls": sum(c["name"] == "get_library_structure" for c in calls),
        "pages_read": sum(page_count(json.loads(c["arguments"] or "{}").get("pages", ""))
                          for c in calls if c["name"] == "get_page_content"),
        "tool_calls": len(calls),
        "latency_s": round(latency, 2),
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0,
        "answer": response["choices"][0]["message"]["content"],
    }


def run_arm(arm, questions, doc_ids, workers):
    path = DATA / f"library-view-{arm}.jsonl"
    rows = [json.loads(line) for line in path.open()] if path.exists() else []
    done = {row["id"] for row in rows if "error" not in row}
    client = make_client(arm)
    lock = threading.Lock()

    def work(question):
        try:
            row = measure(client, question, [doc_ids[f] for f in question["filings"]])
        except Exception as exc:
            row = {"id": question["id"], "error": f"{type(exc).__name__}: {exc}"[:300]}
        with lock, path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(work, [q for q in questions if q["id"] not in done]))
    rows = [json.loads(line) for line in path.open()]
    return list({row["id"]: row for row in rows}.values())


def summarize(results):
    def cost(row):
        uncached = row["input_tokens"] - row["cached_tokens"]
        return (uncached * PRICE["input"] + row["cached_tokens"] * PRICE["cached"]
                + row["output_tokens"] * PRICE["output"]) / 1e6

    metrics = [("structure calls / q", lambda rs: statistics.mean(r["structure_calls"] for r in rs)),
               ("pages read / q", lambda rs: statistics.mean(r["pages_read"] for r in rs)),
               ("tool calls / q", lambda rs: statistics.mean(r["tool_calls"] for r in rs)),
               ("latency median s", lambda rs: statistics.median(r["latency_s"] for r in rs)),
               ("latency mean s", lambda rs: statistics.mean(r["latency_s"] for r in rs)),
               ("input tokens / q", lambda rs: statistics.mean(r["input_tokens"] for r in rs)),
               ("cached tokens / q", lambda rs: statistics.mean(r["cached_tokens"] for r in rs)),
               ("output tokens / q", lambda rs: statistics.mean(r["output_tokens"] for r in rs)),
               ("USD / 1,000 q", lambda rs: 1000 * statistics.mean(cost(r) for r in rs))]
    arms = list(results)
    for label, keep in (("all questions", lambda r: True),
                        ("one filing", lambda r: r["filings"] == 1),
                        ("several filings", lambda r: r["filings"] > 1)):
        groups = {arm: [r for r in results[arm] if "error" not in r and keep(r)] for arm in arms}
        n = {arm: len(rows) for arm, rows in groups.items()}
        print(f"\n{label} (answered: " + ", ".join(f"{a} {n[a]}" for a in arms) + ")")
        print(f"{'':22}" + "".join(f"{arm:>12}" for arm in arms))
        for name, fn in metrics:
            print(f"{name:22}" + "".join(
                f"{fn(groups[arm]):>12,.1f}" if groups[arm] else f"{'-':>12}" for arm in arms))
    for arm in arms:
        errors = [r for r in results[arm] if "error" in r]
        if errors:
            print(f"\n{arm}: {len(errors)} errors, e.g. {errors[0]['error']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=int, default=63)
    parser.add_argument("--arms", default="off,view,topics")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    arms = args.arms.split(",")

    questions = json.loads((HERE / "nvidia-10k-questions.json").read_text())[:args.questions]
    indexer = make_client("off")
    doc_ids = {}
    for path in filings():
        try:
            doc_ids[path.stem] = indexer.get_document_id(path.name)
        except Exception:
            doc_ids[path.stem] = indexer.submit_document(str(path))["doc_id"]
    if "topics" in arms and indexer._api._store.get_library_index() is None:
        built = indexer.build_library_topics()
        print(f"topic index: {built['topics']} topics, "
              f"{built['usage']['total_tokens']:,} tokens")
    summarize({arm: run_arm(arm, questions, doc_ids, args.workers) for arm in arms})


if __name__ == "__main__":
    main()
