from unittest.mock import Mock

import pytest

pytest.importorskip("fastembed")

from rerankers import Reranker
from rerankers.documents import Document
from rerankers.models.fastembed_ranker import FastEmbedRanker


@pytest.fixture
def encoder(monkeypatch):
    model = Mock()
    factory = Mock(return_value=model)
    monkeypatch.setattr("rerankers.models.fastembed_ranker.TextCrossEncoder", factory)
    return factory, model


def test_factory_passes_model_options(encoder):
    factory, _ = encoder
    ranker = Reranker(
        "Xenova/ms-marco-MiniLM-L-6-v2",
        model_type="fastembed",
        verbose=0,
        cache_dir="model-cache",
        threads=2,
    )
    assert isinstance(ranker, FastEmbedRanker)
    factory.assert_called_once_with(
        model_name="Xenova/ms-marco-MiniLM-L-6-v2", cache_dir="model-cache", threads=2
    )


def test_rank_preserves_document_identity_and_raw_scores(encoder):
    _, model = encoder
    model.rerank.return_value = iter([-4.0, 2.5, 2.5])
    ranker = FastEmbedRanker("model", verbose=0, batch_size=2)
    docs = [
        Document("duplicate", doc_id="first", metadata={"source": "a"}),
        Document("duplicate", doc_id="second", metadata={"source": "b"}),
        Document("third", doc_id="third", metadata={"source": "c"}),
    ]
    results = ranker.rank("query", docs)
    assert [r.doc_id for r in results] == ["second", "third", "first"]
    assert [r.score for r in results] == [2.5, 2.5, -4.0]
    assert [r.rank for r in results] == [1, 2, 3]
    assert results[0].document is docs[1]
    assert results[0].metadata == {"source": "b"}
    assert results.query == "query"
    assert results.has_scores
    model.rerank.assert_called_once_with(
        "query", ["duplicate", "duplicate", "third"], batch_size=2
    )


def test_rank_prepares_strings_ids_and_metadata(encoder):
    _, model = encoder
    model.rerank.return_value = iter([0.0, 0.8])
    results = FastEmbedRanker("model", verbose=0).rank(
        "query", ["a", "b"], doc_ids=[9, 4], metadata=[{"i": 1}, {"i": 2}]
    )
    assert [(r.text, r.doc_id, r.metadata) for r in results] == [
        ("b", 4, {"i": 2}),
        ("a", 9, {"i": 1}),
    ]


@pytest.mark.parametrize("doc", ["one", Document("one", doc_id="id")])
def test_rank_single_document(encoder, doc):
    _, model = encoder
    model.rerank.return_value = iter([0.75])
    result = FastEmbedRanker("model", verbose=0).rank("q", doc)
    assert result[0].text == "one"
    assert result[0].score == 0.75


def test_empty_input_does_not_call_encoder(encoder):
    _, model = encoder
    results = FastEmbedRanker("model", verbose=0).rank("query", [])
    assert results.results_count() == 0
    assert results.query == "query"
    assert results.has_scores
    model.rerank.assert_not_called()


def test_score_returns_raw_single_document_score(encoder):
    _, model = encoder
    model.rerank.return_value = iter([-1.25])
    assert FastEmbedRanker("model", verbose=0).score("query", "doc") == -1.25


def test_backend_errors_propagate(encoder):
    _, model = encoder
    model.rerank.side_effect = RuntimeError("backend failed")
    with pytest.raises(RuntimeError, match="backend failed"):
        FastEmbedRanker("model", verbose=0).rank("query", ["doc"])


@pytest.mark.parametrize("scores", [[], [0.2, 0.3]])
def test_score_count_mismatch_is_not_silent(encoder, scores):
    _, model = encoder
    model.rerank.return_value = iter(scores)
    with pytest.raises(ValueError, match="one score per document"):
        FastEmbedRanker("model", verbose=0).rank("query", ["doc"])


@pytest.mark.parametrize("batch_size", [0, -1])
def test_invalid_batch_size(encoder, batch_size):
    with pytest.raises(ValueError, match="batch_size"):
        FastEmbedRanker("model", batch_size=batch_size)


def test_rank_async_uses_same_adapter(encoder):
    import asyncio

    _, model = encoder
    model.rerank.return_value = iter([0.75])
    ranker = FastEmbedRanker("model", verbose=0)
    results = asyncio.run(ranker.rank_async("query", ["doc"], ["id"]))
    assert results[0].doc_id == "id"
    assert results[0].score == 0.75


def test_core_import_and_missing_extra_message():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-S",
            "-c",
            "from rerankers import Reranker; "
            "assert Reranker('model', model_type='fastembed', verbose=0) is None",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert 'pip install "rerankers[fastembed]"' in result.stdout
