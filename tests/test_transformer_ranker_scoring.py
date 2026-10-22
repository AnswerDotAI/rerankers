import pytest
import torch
from tokenizers import Tokenizer, models, pre_tokenizers, processors
from transformers import (
    BertConfig,
    BertForSequenceClassification,
    PreTrainedTokenizerFast,
)

from rerankers.models.transformer_ranker import TransformerRanker


@pytest.fixture(scope="module")
def tiny_rankers(tmp_path_factory):
    root = tmp_path_factory.mktemp("transformer_checkpoints")
    words = [
        "[PAD]",
        "[UNK]",
        "[CLS]",
        "[SEP]",
        "where",
        "is",
        "the",
        "a",
        "red",
        "blue",
        "car",
        "bird",
        "road",
        "hello",
        "world",
    ]
    backend = Tokenizer(
        models.WordPiece({word: i for i, word in enumerate(words)}, unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    backend.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", 2), ("[SEP]", 3)],
    )
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        model_max_length=64,
    )
    rankers = {}
    for labels, name in [(1, "regression"), (2, "monobert")]:
        path = root / name
        torch.manual_seed(labels)
        config = BertConfig(
            vocab_size=len(words),
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            intermediate_size=32,
            num_labels=labels,
            hidden_dropout_prob=0,
            attention_probs_dropout_prob=0,
        )
        BertForSequenceClassification(config).save_pretrained(path)
        tokenizer.save_pretrained(path)
        rankers[labels] = TransformerRanker(
            str(path), device="cpu", dtype=torch.float32, verbose=0
        )
    return rankers


@pytest.mark.parametrize(
    "count,batch_size", [(1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (4, 2), (4, 3), (5, 2)]
)
def test_monobert_ranking_keeps_singleton_batches(tiny_rankers, count, batch_size):
    ranker = tiny_rankers[2]
    query = "where is the red car"
    docs = ["the red car", "a blue bird", "the blue car", "hello world", "a red road"][
        :count
    ]
    ids = [f"doc-{i}" for i in range(count)]
    inputs = ranker.tokenizer([query] * count, docs, padding=True, return_tensors="pt")
    with torch.no_grad():
        logits = ranker.model(**inputs).logits.tolist()
    expected = {doc_id: values[1] - values[0] for doc_id, values in zip(ids, logits)}
    results = ranker.rank(query, docs, doc_ids=ids, batch_size=batch_size)
    assert results.results_count() == count
    assert {result.doc_id for result in results} == set(ids)
    assert [result.score for result in results] == sorted(
        [result.score for result in results], reverse=True
    )
    for result in results:
        assert isinstance(result.score, float)
        assert result.score == pytest.approx(expected[result.doc_id], abs=1e-7)


@pytest.mark.parametrize("labels", [1, 2])
@pytest.mark.parametrize("doc", ["the red car", ""])
def test_transformer_score_uses_one_query_document_pair(tiny_rankers, labels, doc):
    ranker = tiny_rankers[labels]
    query = "where is the red car"
    inputs = ranker.tokenizer([query], [doc], return_tensors="pt")
    with torch.no_grad():
        logits = ranker.model(**inputs).logits[0].tolist()
    expected = logits[1] - logits[0] if labels == 2 else logits[0]
    actual = ranker.score(query, doc)
    assert isinstance(actual, float)
    assert actual == pytest.approx(expected, abs=1e-7)
    assert actual == ranker.rank(query, [doc]).results[0].score
