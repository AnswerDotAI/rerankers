from typing import Union, Optional, List
from pydantic import BaseModel, validator


class Result(BaseModel):
    doc_id: Union[int, str]
    text: str
    score: Optional[float] = None
    rank: Optional[int] = None

    @validator("rank", always=True)
    def check_score_or_rank_exists(cls, v, values):
        if v is None and values.get("score") is None:
            raise ValueError("Either score or rank must be provided.")
        return v


class RankedResults(BaseModel):
    results: List[Result]
    query: str
    has_scores: bool = False

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
        result = next((r for r in self.results if r.doc_id == doc_id), None)
        return result.score if result else None
