from typing import Union, Optional, List
from pydantic import BaseModel, validator

from rerankers.documents import Document


class Result(BaseModel):
    document: Document
    score: Optional[float] = None
    rank: Optional[int] = None

    @validator("rank", always=True)
    def check_score_or_rank_exists(cls, v, values):
        if v is None and values.get("score") is None:
            raise ValueError("Either score or rank must be provided.")
        return v

    def __getattr__(self, item):
        if item in self.document.model_fields:
            return getattr(self.document, item)
        elif item in self.model_fields:
            return getattr(self, item)
        elif item in self.document.metadata:
            return self.document.metadata[item]
        raise AttributeError(
            f"'{self.__class__.__name__}' object has no attribute '{item}'"
        )


class RankedResults(BaseModel):
    results: List[Result]
    query: str
    has_scores: bool = False

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

    def get_result_by_docid(self, doc_id: [Union[int, str]]) -> Result:
        """Fetches a result by its doc_id using a more efficient approach."""
        result = next((r for r in self.results if r.document.doc_id == doc_id), None)
        return result if result else None
