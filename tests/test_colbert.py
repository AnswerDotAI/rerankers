from unittest.mock import patch
from rerankers.models.colbert_ranker import ColBERTRanker
from rerankers.results import Result, RankedResults
from rerankers.documents import Document


@patch("rerankers.models.colbert_ranker.ColBERTRanker.rank")
def test_colbert_ranker_loads_and_ranks(mock_rank):
    # Regression test: ColBERTModel.__init__ must call self.post_init()
    # (not the older self.init_weights()) or model loading raises
    # AttributeError: 'ColBERTModel' object has no attribute
    # 'all_tied_weights_keys' on transformers>=5.
    query = "What is the capital of France?"
    docs = [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
    ]
    expected_results = RankedResults(
        results=[
            Result(
                document=Document(doc_id=0, text="Paris is the capital of France."),
                score=1.0,
                rank=1,
            ),
            Result(
                document=Document(doc_id=1, text="Berlin is the capital of Germany."),
                score=0.5,
                rank=2,
            ),
        ],
        query=query,
        has_scores=True,
    )
    mock_rank.return_value = expected_results

    ranker = ColBERTRanker("colbert-ir/colbertv2.0")
    results = ranker.rank(query=query, docs=docs)

    assert results == expected_results
