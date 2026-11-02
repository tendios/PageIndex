"""Summaries and the description come in the document's language unless a language is forced."""
from __future__ import annotations

import asyncio

import pytest

from pageindex import PageIndexClient
from pageindex import utils


def test_prompts_ask_for_the_language_of_the_text_by_default(monkeypatch):
    seen = []

    async def fake_acompletion(model, prompt, **params):
        seen.append(prompt)
        return "resumen"

    def fake_completion(model, prompt, chat_history=None, return_finish_reason=False):
        seen.append(prompt)
        return "descripción"
    monkeypatch.setattr(utils, "llm_acompletion", fake_acompletion)
    monkeypatch.setattr(utils, "llm_completion", fake_completion)
    asyncio.run(utils.generate_node_summary({"text": "Clàusula 1. Objecte"}, model="openai/x"))
    utils.generate_doc_description([{"title": "Clàusula 1"}], model="openai/x")
    assert len(seen) == 2
    assert all("language of the given text" in prompt and "never translating it into English" in prompt for prompt in seen)


def test_a_forced_language_replaces_the_instruction():
    token = utils._summary_language.set("Spanish")
    try:
        assert utils.summary_language_instruction() == "Write it in Spanish."
    finally:
        utils._summary_language.reset(token)
    assert utils.summary_language_instruction("Catalan") == "Write it in Catalan."


def test_the_client_scopes_the_language_around_indexing(tmp_path, sample_pdf, monkeypatch):
    import pageindex.flash
    prompts = []

    async def fake_acompletion(model, prompt, **params):
        prompts.append(prompt)
        return "resumen"
    monkeypatch.setattr(utils, "llm_acompletion", fake_acompletion)
    monkeypatch.setattr(utils, "llm_completion", lambda model, prompt, **kw: (prompts.append(prompt), "descripción")[1])

    def fake_flash(pdf, **kwargs):
        # flash summarizes inside the library; here the summaries are already written and only the description runs
        return {"structure": [{"title": "Sección 1", "start_index": 1, "end_index": 2, "summary": "s"}], "toc_source": "detected"}
    monkeypatch.setattr(pageindex.flash, "page_index_flash", fake_flash)
    client = PageIndexClient(index={"mode": "local", "model": "openai/x", "storage_path": str(tmp_path / "s"),
                                    "summary_language": "Spanish"})
    client.submit_document(sample_pdf, mode="flash", doc_id="d")
    assert prompts and all("Write it in Spanish." in prompt for prompt in prompts)
    with pytest.raises(Exception):
        PageIndexClient(index={"mode": "local", "model": "openai/x", "storage_path": str(tmp_path / "s2"), "summary_language": 3})
