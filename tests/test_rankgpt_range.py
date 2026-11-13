from types import SimpleNamespace

import pytest

from rerankers.documents import Document

rankgpt_rankers = pytest.importorskip("rerankers.models.rankgpt_rankers")


@pytest.fixture
def ranker_and_requests(monkeypatch):
    requests = []

    def completion(**kwargs):
        requests.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(message=SimpleNamespace(content="[3] > [2] > [1]"))
            ]
        )

    monkeypatch.setattr(rankgpt_rankers, "completion", completion)
    ranker = rankgpt_rankers.RankGPTRanker("offline-test", "unused", verbose=0)
    return ranker, requests


@pytest.mark.parametrize("kwargs", [{}, {"rank_end": 0}])
def test_default_rank_end_includes_all_documents(ranker_and_requests, kwargs):
    ranker, requests = ranker_and_requests
    results = ranker.rank(
        query="query",
        docs=["alpha", "beta", "gamma"],
        doc_ids=["a", "b", "c"],
        metadata=[{"source": "A"}, {"source": "B"}, {"source": "C"}],
        **kwargs,
    )

    messages = requests[0]["messages"]
    assert "3 passages" in messages[1]["content"]
    assert [m["content"] for m in messages if m["content"].startswith("[")] == [
        "[1] alpha",
        "[2] beta",
        "[3] gamma",
    ]
    assert [r.doc_id for r in results] == ["c", "b", "a"]
    assert [r.metadata for r in results] == [
        {"source": "C"},
        {"source": "B"},
        {"source": "A"},
    ]
    assert [r.rank for r in results] == [1, 2, 3]
    assert results.query == "query"
    assert not results.has_scores


def test_default_rank_end_respects_rank_start(ranker_and_requests):
    ranker, requests = ranker_and_requests
    docs = [
        Document(text=text, doc_id=text) for text in ["alpha", "beta", "gamma", "delta"]
    ]
    results = ranker.rank("query", docs, rank_start=1)

    messages = requests[0]["messages"]
    assert "3 passages" in messages[1]["content"]
    assert [m["content"] for m in messages if m["content"].startswith("[")] == [
        "[1] beta",
        "[2] gamma",
        "[3] delta",
    ]
    assert [r.doc_id for r in results] == ["alpha", "delta", "gamma", "beta"]


@pytest.mark.parametrize(
    "start,end,expected_texts,expected_ids",
    [
        (0, 2, ["[1] alpha", "[2] beta"], ["beta", "alpha", "gamma", "delta"]),
        (1, 3, ["[1] beta", "[2] gamma"], ["alpha", "gamma", "beta", "delta"]),
    ],
)
def test_explicit_range_preserves_documents_outside_window(
    ranker_and_requests, start, end, expected_texts, expected_ids
):
    ranker, requests = ranker_and_requests
    docs = [
        Document(text=text, doc_id=text) for text in ["alpha", "beta", "gamma", "delta"]
    ]
    results = ranker.rank("query", docs, rank_start=start, rank_end=end)

    messages = requests[0]["messages"]
    assert "2 passages" in messages[1]["content"]
    assert [
        m["content"] for m in messages if m["content"].startswith("[")
    ] == expected_texts
    assert [r.doc_id for r in results] == expected_ids
