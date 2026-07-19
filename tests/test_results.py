import pytest
from rerankers.results import Result, RankedResults
from rerankers.documents import Document


def test_ranked_results_functions():
    results = RankedResults(
        results=[
            Result(document=Document(doc_id=0, text="Doc 0"), score=0.9, rank=2),
            Result(document=Document(doc_id=1, text="Doc 1"), score=0.95, rank=1),
        ],
        query="Test Query",
        has_scores=True,
    )
    assert results.results_count() == 2
    top_k = results.top_k(1)
    assert len(top_k) == 1
    assert top_k[0].doc_id == 1
    assert results.get_score_by_docid(0) == 0.9


def test_result_attributes():
    result = Result(document=Document(doc_id=1, text="Doc 1"), score=0.95, rank=1)
    assert result.doc_id == 1
    assert result.text == "Doc 1"
    assert result.score == 0.95
    assert result.rank == 1


def test_result_validation_error():
    with pytest.raises(ValueError) as excinfo:
        Result(document=Document(doc_id=2, text="Doc 2"))
    assert "Either score or rank must be provided." in str(excinfo.value)


def test_prep_docs_document_input_autogenerates_ids():
    from rerankers.utils import prep_docs

    # Document objects with no explicit doc_id must receive auto-generated
    # integer ids so get_score_by_docid / get_result_by_docid can resolve them.
    docs = [Document(text="a"), Document(text="b"), Document(text="c")]
    out = prep_docs(docs)
    assert [d.doc_id for d in out] == [0, 1, 2]
