from typing import List, Optional, Union

from rerankers.documents import Document


class Result:
    def __init__(self, document: Document, score: Optional[float] = None, rank: Optional[int] = None):
        self.document = document
        self.score = score
        self.rank = rank

        if rank is None and score is None:
            raise ValueError("Either score or rank must be provided.")

    def __getattr__(self, item):
        if hasattr(self.document, item):
            return getattr(self.document, item)
        elif item in ["document", "score", "rank"]:
            return getattr(self, item)
        elif item in self.document.attributes:
            return getattr(self.document, item)
        elif item in self.document.metadata:
            return self.document.metadata[item]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )

    def __repr__(self) -> str:
        fields = {
            "document": self.document,
            "score": self.score,
            "rank": self.rank,
        }
        field_str = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({field_str})"


class RankedResults:
    def __init__(self, results: List[Result], query: str, has_scores: bool = False):
        self.results = results
        self.query = query
        self.has_scores = has_scores

    def __iter__(self):
        """Allows iteration over the results list."""
        return iter(self.results)

    def __getitem__(self, index):
        """Allows indexing to access results directly."""
        return self.results[index]

    def results_count(self) -> int:
        """Returns the total number of results."""
        return len(self.results)

    def top_k(self, k: int) -> List[Result]:
        """Returns the top k results based on the score, if available, or rank."""
        if self.has_scores:
            return sorted(
                self.results,
                key=lambda x: x.score if x.score is not None else float("-inf"),
                reverse=True,
            )[:k]
        else:
            return sorted(
                self.results,
                key=lambda x: x.rank if x.rank is not None else float("inf"),
            )[:k]

    def get_score_by_docid(self, doc_id: Union[int, str]) -> Optional[float]:
        """Fetches the score of a result by its doc_id using a more efficient approach."""
        result = next((r for r in self.results if r.document.doc_id == doc_id), None)
        return result.score if result else None

    def get_result_by_docid(self, doc_id: Union[int, str]) -> Optional[Result]:
        """Fetches a result by its doc_id using a more efficient approach."""
        result = next((r for r in self.results if r.document.doc_id == doc_id), None)
        return result if result else None

    def __repr__(self) -> str:
        fields = {
            "results": self.results,
            "query": self.query,
            "has_scores": self.has_scores,
        }
        field_str = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.__class__.__name__}({field_str})"
