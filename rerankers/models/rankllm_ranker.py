from typing import Optional, Union, List
from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from rerankers.results import RankedResults, Result
from rerankers.utils import prep_docs

from rank_llm.data import Candidate, Query, Request
from rank_llm.rerank.vicuna_reranker import VicunaReranker
from rank_llm.rerank.zephyr_reranker import ZephyrReranker
from rank_llm.rerank.rank_gpt import SafeOpenai
from rank_llm.rerank.reranker import Reranker as rankllm_Reranker


class RankLLMRanker(BaseRanker):
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        lang: str = "en",
        verbose: int = 1,
    ) -> "RankLLMRanker":
        self.api_key = api_key
        self.model = model
        self.verbose = verbose
        self.lang = lang

        if "zephyr" in self.model.lower():
            self.rankllm_ranker = ZephyrReranker()
        elif "vicuna" in self.model.lower():
            self.rankllm_ranker = VicunaReranker()
        elif "gpt" in self.model.lower():
            self.rankllm_ranker = rankllm_Reranker(
                SafeOpenai(model=self.model, context_size=4096, keys=self.api_key)
            )

    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
        rank_start: int = 0,
        rank_end: int = 0,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)

        request = Request(
            query=Query(text=query, qid=1),
            candidates=[
                Candidate(doc={"text": doc.text}, docid=doc_idx, score=1)
                for doc_idx, doc in enumerate(docs)
            ],
        )

        rankllm_results = self.rankllm_ranker.rerank(
            request,
            rank_end=len(docs) if rank_end == 0 else rank_end,
            window_size=min(20, len(docs)),
            step=10,
        )

        ranked_docs = []

        for rank, result in enumerate(rankllm_results.candidates, start=rank_start):
            ranked_docs.append(
                Result(
                    document=docs[result.docid],
                    rank=rank,
                )
            )

        return RankedResults(results=ranked_docs, query=query, has_scores=False)

    def score(self):
        print("Listwise ranking models like RankLLM cannot output scores!")
        return None
