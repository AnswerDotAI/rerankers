from rerankers.models.ranker import BaseRanker

import torch
from typing import Union, List, Optional, Tuple
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
)
from rerankers.utils import (
    vprint,
    get_device,
    get_dtype,
    ensure_docids,
    ensure_docs_list,
)
from rerankers.results import RankedResults, Result


class TransformerRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 16,
        verbose: int = 1,
    ):
        self.verbose = verbose
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path, torch_dtype=self.dtype
        ).to(self.device)
        vprint(f"Loaded model {model_name_or_path}", self.verbose)
        vprint(f"Using device {self.device}.", self.verbose)
        vprint(f"Using dtype {self.dtype}.", self.verbose)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        self.ranking_type = "pointwise"
        self.batch_size = batch_size

    def tokenize(self, inputs: Union[str, List[str], List[Tuple[str, str]]]):
        return self.tokenizer(
            inputs, return_tensors="pt", padding=True, truncation=True
        ).to(self.device)

    @torch.no_grad()
    def rank(
        self,
        query: str,
        docs: List[str],
        doc_ids: Optional[List[Union[str, int]]] = None,
        batch_size: Optional[int] = None,
    ) -> RankedResults:
        docs = ensure_docs_list(docs)
        doc_ids = ensure_docids(doc_ids, len(docs))
        inputs = [(query, doc) for doc in docs]

        # Override self.batch_size if explicitely set
        if batch_size is None:
            batch_size = self.batch_size
        batched_inputs = [
            inputs[i : i + batch_size] for i in range(0, len(inputs), batch_size)
        ]
        scores = []
        for batch in batched_inputs:
            tokenized_inputs = self.tokenize(batch)
            batch_scores = self.model(**tokenized_inputs).logits.squeeze()
            batch_scores = batch_scores.detach().cpu().numpy().tolist()
            if isinstance(batch_scores, float):  # Handling the case of single score
                scores.append(batch_scores)
            else:
                scores.extend(batch_scores)
        if len(scores) == 1:
            return Result(doc_id=doc_ids[0], text=docs[0], score=scores[0])
        else:
            ranked_results = [
                Result(doc_id=doc_id, text=doc, score=score, rank=idx + 1)
                for idx, (doc_id, doc, score) in enumerate(
                    sorted(zip(doc_ids, docs, scores), key=lambda x: x[2], reverse=True)
                )
            ]
            return RankedResults(results=ranked_results, query=query, has_scores=True)

    @torch.no_grad()
    def score(self, query: str, doc: str) -> float:
        inputs = self.tokenize((query, doc))
        outputs = self.model(**inputs)
        score = outputs.logits.squeeze().detach().cpu().numpy().astype(float)
        return score
