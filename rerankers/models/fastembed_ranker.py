from typing import List, Optional, Union

from fastembed.rerank.cross_encoder import TextCrossEncoder

from rerankers.documents import Document
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result
from rerankers.utils import prep_docs, vprint


class FastEmbedRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        verbose: int = 1,
        batch_size: int = 64,
        **kwargs,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.verbose = verbose
        self.batch_size = batch_size
        self.ranking_type = "pointwise"
        vprint(f"Loading FastEmbed model {model_name_or_path}...", verbose)
        self.model = TextCrossEncoder(model_name=model_name_or_path, **kwargs)

    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        if isinstance(docs, list) and not docs:
            return RankedResults(results=[], query=query, has_scores=True)
        docs = prep_docs(docs, doc_ids, metadata)
        scores = list(
            self.model.rerank(
                query, [doc.text for doc in docs], batch_size=self.batch_size
            )
        )
        if len(scores) != len(docs):
            raise ValueError("FastEmbed must return one score per document")
        scored_docs = sorted(zip(docs, scores), key=lambda pair: pair[1], reverse=True)
        results = [
            Result(document=doc, score=float(score), rank=index + 1)
            for index, (doc, score) in enumerate(scored_docs)
        ]
        return RankedResults(results=results, query=query, has_scores=True)

    def score(self, query: str, doc: str) -> float:
        return self.rank(query, doc)[0].score
