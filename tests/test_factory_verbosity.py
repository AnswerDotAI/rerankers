from unittest.mock import Mock, sentinel

import pytest
from rerankers import Reranker
from rerankers.reranker import AVAILABLE_RANKERS, DEFAULTS


@pytest.mark.parametrize(
    "model_name,model_type",
    [
        ("colbert", "ColBERTRanker"),
        ("t5", "T5Ranker"),
        ("cohere", "APIRanker"),
        ("jina", "APIRanker"),
    ],
)
@pytest.mark.parametrize("verbose", [0, 1])
def test_default_model_loading_respects_verbosity(
    monkeypatch, capsys, model_name, model_type, verbose
):
    constructor = Mock(return_value=sentinel.ranker)
    monkeypatch.setitem(AVAILABLE_RANKERS, model_type, constructor)

    ranker = Reranker(model_name, verbose=verbose)

    assert ranker is sentinel.ranker
    assert constructor.call_args.args == (DEFAULTS[model_name]["en"],)
    assert constructor.call_args.kwargs["verbose"] == verbose
    output = capsys.readouterr().out
    if verbose == 0:
        assert output == ""
    else:
        assert DEFAULTS[model_name]["en"] in output


def test_missing_dependencies_still_reported_when_quiet(monkeypatch, capsys):
    monkeypatch.delitem(AVAILABLE_RANKERS, "ColBERTRanker", raising=False)

    assert Reranker("colbert", verbose=0) is None

    output = capsys.readouterr().out
    assert "necessary dependencies" in output
    assert "rerankers[transformers]" in output
    assert "Loading default" not in output
