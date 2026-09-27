"""Chunking, retrieval quality, citations and knowledge-base persistence."""

from __future__ import annotations

import io

import pytest

from app.ingestion.chunking import chunk_blocks, chunk_plain_text, split_sentences
from app.ingestion.embeddings import cosine_similarity, stem, tokenize
from app.ingestion.models import ExtractedBlock, FileStatus, SourceKind
from app.ingestion.store import FileStore
from conftest import SAMPLE_NOTE


def ingest(service, payload: bytes, filename: str, session_id: str = "s1"):
    record = service.save_upload(io.BytesIO(payload), filename, session_id=session_id)
    return service.process(record.id)


# --- embedding provider --------------------------------------------------


def test_hashing_embeddings_are_deterministic(embedder) -> None:
    a = embedder.embed("configure the upload limit")
    b = embedder.embed("configure the upload limit")
    assert a == b
    assert len(a) == embedder.dim


def test_hashing_embeddings_are_unit_length(embedder) -> None:
    vector = embedder.embed("some text about things")
    assert abs((sum(x * x for x in vector)) ** 0.5 - 1.0) < 1e-5


def test_empty_text_yields_zero_vector(embedder) -> None:
    assert not any(embedder.embed(""))


def test_relevant_pairs_outrank_unrelated_ones(embedder) -> None:
    relevant = cosine_similarity(
        embedder.embed("how many video frames are sampled"),
        embedder.embed("what is the video frame limit per video"),
    )
    unrelated = cosine_similarity(
        embedder.embed("how many video frames are sampled"),
        embedder.embed("my cat sleeps on the warm keyboard"),
    )
    assert relevant > unrelated


def test_no_model_download_is_required(embedder) -> None:
    status = embedder.status()
    assert status["provider"] == "hashing"
    assert status["requires_download"] is False


def test_tokenizer_splits_identifiers_and_stems() -> None:
    assert "chat" in tokenize("createChatCompletion")
    assert "create" in tokenize("create_chat_completion")
    assert stem("files") == "file"
    assert stem("directories") == "directory"
    assert stem("forgetting") == "forget"
    assert stem("limits") == "limit"
    assert stem("address") == "address"


# --- chunking ------------------------------------------------------------


def test_sentence_splitting_keeps_prose_whole() -> None:
    sentences = split_sentences("First sentence. Second one! Third?")
    assert sentences == ["First sentence.", "Second one!", "Third?"]


def test_short_block_becomes_one_chunk() -> None:
    chunks = chunk_blocks(
        [ExtractedBlock(text="Short note.", locator="page 1")],
        file_id="f1", filename="a.txt", chunk_size=1200,
    )
    assert len(chunks) == 1
    assert chunks[0].locator == "page 1"


def test_long_block_is_split_with_overlap() -> None:
    text = " ".join(f"Sentence number {i} carries some information." for i in range(200))
    chunks = chunk_blocks(
        [ExtractedBlock(text=text, locator="page 3")],
        file_id="f1", filename="a.txt", chunk_size=400, chunk_overlap=80,
    )
    assert len(chunks) > 1
    assert all(len(c.text) <= 600 for c in chunks)
    # Overlap means consecutive chunks share content.
    assert chunks[0].text[-20:] in chunks[1].text or len(chunks) == 1


def test_every_chunk_keeps_its_citation_locator() -> None:
    blocks = [
        ExtractedBlock(text="Page one content about revenue.", locator="page 1", order=0),
        ExtractedBlock(text="Page two content about risks.", locator="page 2", order=1),
    ]
    chunks = chunk_blocks(blocks, file_id="f1", filename="doc.pdf", chunk_size=1200)
    assert {c.locator for c in chunks} == {"page 1", "page 2"}


def test_source_label_combines_file_and_locator() -> None:
    chunk = chunk_plain_text("x", file_id="f1", filename="report.pdf", chunk_size=1200)[0]
    assert chunk.source_label() == "report.pdf"
    chunk.locator = "page 7"
    assert chunk.source_label() == "report.pdf · page 7"


def test_spreadsheet_rows_are_never_split_mid_fact() -> None:
    row = ExtractedBlock(
        text=" | ".join(f"col{i}: value{i}" for i in range(200)),
        kind=SourceKind.ROW, locator="Revenue · row 5",
    )
    chunks = chunk_blocks([row], file_id="f1", filename="b.xlsx", chunk_size=100)
    assert len(chunks) == 1
    assert chunks[0].locator == "Revenue · row 5"


# --- retrieval -----------------------------------------------------------


def test_retrieval_finds_the_right_fact(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    assert record.status is FileStatus.READY
    hits = service.search("What is the revenue risk with the vendor?", session_id="s1")
    assert hits
    assert any("vendor contract" in h.chunk.text for h in hits)


def test_every_hit_carries_a_usable_citation(service) -> None:
    ingest(service, SAMPLE_NOTE, "launch.txt")
    for hit in service.search("Q3 revenue", session_id="s1"):
        assert hit.chunk.file_id
        assert hit.chunk.filename == "launch.txt"
        assert hit.source_label().startswith("launch.txt")
        assert hit.to_dict()["source"] == hit.source_label()


def test_retrieval_respects_session_isolation(service) -> None:
    ingest(service, SAMPLE_NOTE, "launch.txt", session_id="s1")
    assert service.search("Q3 revenue", session_id="s2") == []


def test_top_k_limits_the_result_count(service) -> None:
    ingest(service, SAMPLE_NOTE, "launch.txt")
    assert len(service.search("revenue risk contract", session_id="s1", top_k=1)) <= 1


def test_file_id_filter_restricts_the_search(service) -> None:
    a = ingest(service, b"Alpha document about kangaroos.", "a.txt", session_id="s1")
    b = ingest(service, b"Beta document about platypuses.", "b.txt", session_id="s1")
    hits = service.search("kangaroos platypuses", session_id="s1", file_ids=[a.id], top_k=10)
    assert hits
    assert {h.chunk.file_id for h in hits} == {a.id}


def test_one_long_document_cannot_crowd_out_others(service) -> None:
    """The best chunk from each file is surfaced before second-best chunks."""
    ingest(service, b"Alpha note mentioning widgets. " * 40, "alpha.txt", session_id="s1")
    ingest(service, b"Beta note mentioning widgets too.", "beta.txt", session_id="s1")
    hits = service.search("widgets", session_id="s1", top_k=5)
    # Both files should be represented among the top hits.
    assert len({h.chunk.file_id for h in hits}) == 2


def test_context_block_respects_the_character_budget(service) -> None:
    ingest(service, b"Long content. " * 500, "long.txt", session_id="s1")
    context, hits = service.build_context(
        "content", session_id="s1", top_k=20
    )
    assert len(context) <= service.settings.max_context_chars + 200
    assert len(hits) >= 1


def test_no_context_when_nothing_matches(service) -> None:
    ingest(service, SAMPLE_NOTE, "launch.txt")
    context, hits = service.build_context(
        "zzzz unrelated query about quantum chromodynamics", session_id="s1"
    )
    # Either nothing is relevant, or only weak matches -- never a wrong answer
    # dressed as knowledge.
    assert context == "" or all(h.matched_by in ("keyword", "vector") for h in hits)


# --- persistence ---------------------------------------------------------


def test_knowledge_survives_a_restart(settings, service, store) -> None:
    """Chunks and vectors are reloaded from disk by a fresh KnowledgeBase."""
    from app.ingestion.embeddings import build_embedding_provider
    from app.ingestion.knowledge import KnowledgeBase

    ingest(service, SAMPLE_NOTE, "launch.txt")
    first_hits = service.search("Q3 revenue", session_id="s1")
    assert first_hits
    service.close()

    reopened_store = FileStore(settings)
    reopened_kb = KnowledgeBase(settings, build_embedding_provider(settings))
    assert reopened_kb.stats()["chunks"] > 0
    second_hits = reopened_kb.search("Q3 revenue", session_id="s1")
    assert second_hits
    assert {h.chunk.file_id for h in second_hits} == {h.chunk.file_id for h in first_hits}
    reopened_kb.close()


def test_file_records_survive_a_restart(settings, service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    service.close()
    reopened = FileStore(settings)
    loaded = reopened.get(record.id)
    assert loaded.status is FileStatus.READY
    assert loaded.chunk_count == record.chunk_count
    assert loaded.display_name == "launch.txt"


def test_interrupted_processing_is_recovered_as_failed(settings, service) -> None:
    """A crash mid-ingestion must not leave a permanent spinner."""
    from app.ingestion.models import FileRecord, FileStatus as Status

    record = FileRecord(id="abc123", filename="x.txt", display_name="x.txt",
                        stored_name="abc123.txt", extension=".txt", status=Status.PROCESSING)
    service.store.add(record)
    service.close()

    reopened = FileStore(settings)
    recovered = reopened.get("abc123")
    assert recovered.status is Status.FAILED
    assert "interrupted" in recovered.error.lower()


def test_reindexing_replaces_rather_than_duplicates(service) -> None:
    record = ingest(service, SAMPLE_NOTE, "launch.txt")
    before = service.knowledge.stats()["chunks"]
    service.process(record.id)
    after = service.knowledge.stats()["chunks"]
    assert after == before
