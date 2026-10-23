import json
from unittest.mock import Mock, patch

import pytest

from rerankers.documents import Document
from rerankers.models.api_rankers import APIRanker

PROVIDERS = [
    ("cohere", "results", "relevance_score", "documents"),
    ("jina", "results", "relevance_score", "documents"),
    ("isaacus", "results", "score", "texts"),
    ("voyage", "data", "relevance_score", "documents"),
    ("mixedbread.ai", "data", "score", "input"),
    ("pinecone", "data", "score", "documents"),
    ("text-embeddings-inference", None, "score", "texts"),
]


@pytest.mark.parametrize("provider,results_key,score_key,documents_key", PROVIDERS)
@pytest.mark.parametrize("single_score", [True, False], ids=["score", "rank-control"])
def test_api_ranker_single_document(
    provider, results_key, score_key, documents_key, single_score
):
    ranker = APIRanker(
        model="test-model",
        api_key="test-placeholder",
        api_provider=provider,
        url="https://example.invalid/rerank",
        verbose=0,
    )
    rows = [{"index": 0, score_key: 0.75}]
    response = Mock()
    response.json.return_value = {results_key: rows} if results_key else rows

    with patch(
        "rerankers.models.api_rankers.requests.post", return_value=response
    ) as post:
        if single_score:
            assert ranker.score("search query", "candidate passage") == 0.75
        else:
            results = ranker.rank("search query", "candidate passage")
            assert results.results[0].score == 0.75
            assert isinstance(results.results[0].document, Document)
            assert results.results[0].text == "candidate passage"

    post.assert_called_once()
    payload = json.loads(post.call_args.kwargs["data"])
    expected_documents = (
        [{"text": "candidate passage"}]
        if provider == "pinecone"
        else ["candidate passage"]
    )
    assert payload[documents_key] == expected_documents
    assert payload["query"] == "search query"
    top_key = "top_k" if provider in ("voyage", "mixedbread.ai") else "top_n"
    assert payload[top_key] == 1
