"""Offline ColBERT regression tests using a real, locally initialized BERT."""
import weakref

import pytest
import torch
from transformers import BertConfig, BertTokenizerFast

from rerankers.documents import Document
from rerankers.models import colbert_ranker
from rerankers.models.colbert_ranker import ColBERTModel, ColBERTRanker


@pytest.fixture(scope="module")
def model_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("colbert")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
             "[unused0]", "[unused1]", "apple", "pear", "banana", "."]
    (path / "vocab.txt").write_text("\n".join(vocab))
    BertTokenizerFast(vocab_file=str(path / "vocab.txt")).save_pretrained(path)
    with torch.random.fork_rng():
        torch.manual_seed(7)
        model = ColBERTModel(BertConfig(
            vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=32,
            max_position_embeddings=128,
        ), verbose=0)
    model.save_pretrained(path)
    return str(path)


@pytest.fixture
def ranker(model_path):
    return ColBERTRanker(model_path, device="cpu", dtype="float32", verbose=0)


def reference_scores(ranker, query, docs):
    """One document at a time, with a fixed token budget and explicit MaxSim."""
    with torch.inference_mode():
        query_encoding = ranker._query_encode([query])
        query_reps = ranker.model(**query_encoding)[0]
        query_reps = torch.nn.functional.normalize(query_reps, dim=-1)
        query_reps = query_reps[query_encoding["attention_mask"][0].bool()]
        scores = []
        for doc in docs:
            ids = ranker.tokenizer.encode(
                doc, max_length=ranker.doc_max_length - 1, truncation=True
            )
            ids.insert(1, ranker.document_token_id)
            input_ids = torch.tensor([ids])
            reps = ranker.model(input_ids=input_ids,
                                attention_mask=torch.ones_like(input_ids))[0]
            reps = torch.nn.functional.normalize(reps, dim=-1)
            scores.append((query_reps @ reps.T).max(dim=-1).values.mean().item())
    return scores


@pytest.mark.parametrize("length", [29, 30, 31, 61, 62, 63, 93, 94, 95, 123, 124, 150])
def test_document_tokens_independent_of_batch(ranker, length):
    doc = " ".join(["apple"] * length)
    expected = ranker.tokenizer.encode(
        doc, max_length=ranker.doc_max_length - 1, truncation=True
    )
    expected.insert(1, ranker.document_token_id)
    for docs in ([doc], [doc, " ".join(["pear"] * 150)]):
        encoding = ranker._document_encode(docs)
        actual = encoding["input_ids"][0][encoding["attention_mask"][0].bool()]
        assert actual.tolist() == expected
        assert encoding["input_ids"].shape[1] <= ranker.doc_max_length


@pytest.mark.parametrize("batch_size", [1, 2, 3, 32])
@pytest.mark.parametrize("query", ["apple banana", " ".join(["apple"] * 35)])
def test_scores_and_results_match_reference(ranker, batch_size, query):
    ranker.batch_size = batch_size
    texts = ["", "pear", " ".join(["apple"] * 29 + ["banana"]),
             " ".join(["pear"] * 62), " ".join(["banana"] * 150), "pear"]
    expected = reference_scores(ranker, query, texts)
    actual = ranker._colbert_rank(query, texts)
    assert actual == pytest.approx(expected, abs=2e-6)
    ids = [f"doc-{i}" for i in range(len(texts))]
    metadata = [{"source": i} for i in range(len(texts))]
    results = ranker.rank(query, texts, doc_ids=ids, metadata=metadata)
    order = sorted(range(len(texts)), key=lambda i: actual[i], reverse=True)
    assert [result.doc_id for result in results.results] == [ids[i] for i in order]
    for rank, (result, i) in enumerate(zip(results.results, order), 1):
        assert result.rank == rank
        assert result.text == texts[i]
        assert result.document.metadata == metadata[i]
        assert result.score == pytest.approx(expected[i], abs=2e-6)
    documents = [Document(text=t, doc_id=i, metadata=m)
                 for t, i, m in zip(texts, ids, metadata)]
    assert [r.doc_id for r in ranker.rank(query, documents).results] == [ids[i] for i in order]
    assert ranker.score(query, texts[2]) == pytest.approx(expected[2], abs=2e-6)


def test_document_intermediates_are_bounded_and_released(ranker, monkeypatch):
    ranker.batch_size = 3
    encode = ranker._document_encode
    score = colbert_ranker._colbert_score
    refs = []
    batches = []
    query_calls = []
    query_encode = ranker._query_encode

    def checked_encode(docs):
        assert all(ref() is None for ref in refs)
        assert len(docs) <= ranker.batch_size
        batches.append(len(docs))
        result = encode(docs)
        refs.extend(weakref.ref(value) for value in result.values())
        return result

    def checked_score(q_reps, p_reps, q_mask, p_mask):
        assert p_reps.shape[0] <= ranker.batch_size
        assert p_mask.shape[0] <= ranker.batch_size
        refs.append(weakref.ref(p_reps))
        return score(q_reps, p_reps, q_mask, p_mask)

    def checked_query(query):
        query_calls.append(query)
        return query_encode(query)

    monkeypatch.setattr(ranker, "_document_encode", checked_encode)
    monkeypatch.setattr(colbert_ranker, "_colbert_score", checked_score)
    monkeypatch.setattr(ranker, "_query_encode", checked_query)
    assert len(ranker._colbert_rank("apple", ["pear"] * 11)) == 11
    assert batches == [3, 3, 3, 2]
    assert len(query_calls) == 1
    assert all(ref() is None for ref in refs)


def test_empty_documents(ranker):
    assert ranker._colbert_rank("apple", []) == []
