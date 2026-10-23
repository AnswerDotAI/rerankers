import pytest
import torch

from rerankers.documents import Document

PyLateRanker = pytest.importorskip("rerankers.models.pylate_ranker").PyLateRanker


class FixedEncoder:
    """Replace model inference while exercising PyLate's real MaxSim ranking."""

    def encode(self, texts, is_query):
        vectors = {
            "query": [[1.0, 0.0], [0.0, 1.0]],
            "low": [[-1.0, 0.0], [0.0, -1.0]],
            "middle": [[1.0, 0.0]],
            "high": [[1.0, 0.0], [0.0, 1.0]],
        }
        if isinstance(texts, str):
            return torch.tensor(vectors[texts])
        if texts and isinstance(texts[0], list):
            return [self.encode(batch, is_query) for batch in texts]
        return [torch.tensor(vectors[text]) for text in texts]


@pytest.fixture
def ranker():
    ranker = object.__new__(PyLateRanker)
    ranker.model = FixedEncoder()
    return ranker


@pytest.mark.parametrize(
    "doc_ids", [None, ["a", "b", "c"], [10, 20, 30], ["same", "same", "c"]]
)
def test_rank_keeps_scores_and_metadata_with_the_sorted_document(ranker, doc_ids):
    metadata = [{"source": i} for i in range(3)]
    result = ranker.rank(
        "query", ["low", "high", "middle"], doc_ids=doc_ids, metadata=metadata
    )
    ids = doc_ids if doc_ids is not None else [0, 1, 2]

    assert [item.text for item in result.results] == ["high", "middle", "low"]
    assert [item.doc_id for item in result.results] == [ids[1], ids[2], ids[0]]
    assert [item.metadata for item in result.results] == [
        metadata[1],
        metadata[2],
        metadata[0],
    ]
    assert [item.score for item in result.results] == pytest.approx([1.0, 0.5, 0.0])
    assert [item.rank for item in result.results] == [1, 2, 3]
    assert result.top_k(1)[0].text == "high"


def test_rank_preserves_document_objects_without_separate_ids(ranker):
    docs = [Document("low", doc_id="low-id"), Document("high", doc_id="high-id")]
    result = ranker.rank("query", docs)

    assert result.results[0].document is docs[1]
    assert result.results[1].document is docs[0]
    assert [item.score for item in result.results] == pytest.approx([1.0, 0.0])


@pytest.mark.parametrize("doc", ["high", Document("high", doc_id="custom")])
def test_rank_accepts_a_single_document(ranker, doc):
    result = ranker.rank("query", doc)

    assert len(result.results) == 1
    assert result.results[0].text == "high"
    assert result.results[0].score == pytest.approx(1.0)


@pytest.mark.parametrize(
    "doc, expected", [("low", 0.0), ("middle", 0.5), ("high", 1.0)]
)
def test_score_uses_the_same_query_normalized_maxsim_as_rank(ranker, doc, expected):
    score = ranker.score("query", doc)

    assert isinstance(score, float)
    assert score == pytest.approx(expected)
    assert score == pytest.approx(ranker.rank("query", [doc]).results[0].score)


def test_rank_already_sorted_documents_is_unchanged(ranker):
    result = ranker.rank("query", ["high", "middle", "low"], doc_ids=[1, 2, 3])

    assert [item.text for item in result.results] == ["high", "middle", "low"]
    assert [item.score for item in result.results] == pytest.approx([1.0, 0.5, 0.0])
