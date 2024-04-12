from rerankers.models.ranker import BaseRanker

from flashrank import Ranker, RerankRequest


from typing import Union, List, Optional, Tuple
from rerankers.utils import (
    vprint,
    ensure_docids,
    ensure_docs_list,
)
from rerankers.results import RankedResults, Result


class FlashRankRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        verbose: int = 1,
        cache_dir: str = "./.flashrank_cache",
    ):
        self.verbose = verbose
        vprint(
            f"Loading model FlashRank model {model_name_or_path}...", verbose=verbose
        )
        self.model = Ranker(model_name=model_name_or_path, cache_dir=cache_dir)
        self.ranking_type = "pointwise"

    def tokenize(self, inputs: Union[str, List[str], List[Tuple[str, str]]]):
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

    def rank(
        self,
        query: str,
        docs: List[str],
        doc_ids: Optional[List[Union[str, int]]] = None,
    ) -> RankedResults:
        docs = ensure_docs_list(docs)
        doc_ids = ensure_docids(doc_ids, len(docs))
        passages = [{"id": doc_id, "text": doc} for doc_id, doc in zip(doc_ids, docs)]

        rerank_request = RerankRequest(query=query, passages=passages)
        flashrank_results = self.model.rerank(rerank_request)

        ranked_results = [
            Result(
                doc_id=result["id"],
                text=result["text"],
                score=result["score"],
                rank=idx + 1,
            )
            for idx, result in enumerate(flashrank_results)
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def score(self, query: str, doc: str) -> float:
        rerank_request = RerankRequest(
            query=query, passages=[{"id": "temp_id", "text": doc}]
        )
        flashrank_result = self.model.rerank(rerank_request)
        score = flashrank_result[0]["score"]
        return score
