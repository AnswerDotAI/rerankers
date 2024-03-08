from unittest.mock import patch
import torch
from rerankers import Reranker
from rerankers.models.transformer_ranker import TransformerRanker
from rerankers.results import Result, RankedResults


@patch("rerankers.models.transformer_ranker.TransformerRanker.rank")
def test_transformer_ranker_rank(mock_rank):
    query = "Gone with the wind is an absolute masterpiece"
    docs = [
        "Gone with the wind is a masterclass in bad storytelling.",
        "Gone with the wind is an all-time classic",
    ]
    expected_results = RankedResults(
        results=[
            Result(
                doc_id=1,
                text="Gone with the wind is an all-time classic",
                score=1.6181640625,
                rank=1,
            ),
            Result(
                doc_id=0,
                text="Gone with the wind is a masterclass in bad storytelling.",
                score=0.88427734375,
                rank=2,
            ),
        ],
        query=query,
        has_scores=True,
    )
    mock_rank.return_value = expected_results
    ranker = TransformerRanker("mixedbread-ai/mxbai-rerank-xsmall-v1")
    results = ranker.rank(query=query, docs=docs)
    assert results == expected_results
