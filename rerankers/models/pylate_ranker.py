from typing import List, Optional, Union

import torch

from pylate import models, rank
from rerankers.documents import Document
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result
from rerankers.utils import get_device, get_dtype, prep_docs, vprint


class PyLateRanker(BaseRanker):
    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: int = 1,
        **kwargs,
    ):
        self.verbose = verbose
        self.device = get_device(device, self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size
        vprint(
            f"Loading model {model_name}, this might take a while...",
            self.verbose,
        )
        kwargs = kwargs.get("kwargs", {})
        kwargs["device"] = self.device
        model_kwargs = kwargs.get("model_kwargs", {})
        model_kwargs["torch_dtype"] = self.dtype
        self.model = models.ColBERT(
            model_name_or_path=model_name,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    def rank(
        self,
        query: str,
        docs: Union[Document, str, List[Document], List[str]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)
        documents_embeddings = self.model.encode(
            [[d.text for d in docs]],
            is_query=False,
        )

        query_embeddings = self.model.encode(
            [query],
            is_query=True,
        )
        scores = rank.rerank(
            # Use positions so missing or duplicate external IDs stay unambiguous.
            documents_ids=[list(range(len(docs)))],
            queries_embeddings=query_embeddings,
            documents_embeddings=documents_embeddings,
        )

        ranked_results = [
            Result(
                document=docs[score["id"]],
                score=score["score"] / len(query_embeddings[0]),
                rank=idx + 1,
            )
            for idx, score in enumerate(scores[0])
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def score(self, query: str, doc: str) -> float:
        return self.rank(query, [doc]).results[0].score
