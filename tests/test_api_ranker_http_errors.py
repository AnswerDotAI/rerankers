import json
from unittest.mock import patch

import pytest
import requests

from rerankers.documents import Document
from rerankers.models.api_rankers import APIRanker

PROVIDERS = [
    ("cohere", "results", "relevance_score"),
    ("jina", "results", "relevance_score"),
    ("isaacus", "results", "score"),
    ("voyage", "data", "relevance_score"),
    ("mixedbread.ai", "data", "score"),
    ("pinecone", "data", "score"),
    ("text-embeddings-inference", None, "score"),
]


@pytest.mark.parametrize("provider,results_key,score_key", PROVIDERS)
@pytest.mark.parametrize(
    "status,body",
    [
        (401, b'{"message":"Invalid API key"}'),
        (429, b'{"message":"Rate limited"}'),
        (503, b"Service unavailable"),
    ],
)
def test_rank_preserves_http_failure(provider, results_key, score_key, status, body):
    ranker = APIRanker(
        "model",
        "placeholder",
        provider,
        verbose=0,
        url="https://example.invalid/rerank",
    )
    response = requests.Response()
    response.status_code = status
    response.url = ranker.url
    response._content = body
    response.headers["Retry-After"] = "30"

    with patch(
        "rerankers.models.api_rankers.requests.post", return_value=response
    ), pytest.raises(requests.HTTPError) as error:
        ranker.rank("question", ["passage"])

    assert error.value.response is response
    assert error.value.response.status_code == status
    assert error.value.response.headers["Retry-After"] == "30"
    assert error.value.response.content == body


@pytest.mark.parametrize("provider,results_key,score_key", PROVIDERS)
def test_rank_still_maps_successful_response(provider, results_key, score_key):
    ranker = APIRanker(
        "model",
        "placeholder",
        provider,
        verbose=0,
        url="https://example.invalid/rerank",
    )
    docs = [Document(text="first", doc_id="a"), Document(text="second", doc_id="b")]
    rows = [{"index": 1, score_key: 0.9}, {"index": 0, score_key: 0.2}]
    response = requests.Response()
    response.status_code = 200
    response.url = ranker.url
    response._content = json.dumps(
        {results_key: rows} if results_key else rows
    ).encode()
    with patch("rerankers.models.api_rankers.requests.post", return_value=response):
        result = ranker.rank("question", docs)

    assert result.query == "question"
    assert result.has_scores
    assert [r.doc_id for r in result] == ["b", "a"]
    assert [r.score for r in result] == [0.9, 0.2]
    assert [r.rank for r in result] == [1, 2]
