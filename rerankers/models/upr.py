# models/upr.py

import torch
import torch.nn.functional as F
from transformers import T5ForConditionalGeneration, T5Tokenizer
from math import ceil
from typing import List, Optional, Union

from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from rerankers.results import RankedResults, Result
from rerankers.utils import (
    vprint,
    get_device,
    get_dtype,
    prep_docs,
    get_chunks,
)


class UPRRanker(BaseRanker):
    """
    UPR (Unsupervised Passage Reranker) replicates the negative log-likelihood
    approach from the authors' code. The doc is passed as the encoder input,
    and the query is the decoder label.
    """

    def __init__(
        self,
        model_name_or_path: str,
        verbose: int = 1,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: int = 16,
        verbalizer_head: str = "Passage:",
        verbalizer: str = "Please write a question based on this passage.",
        max_input_length: int = 512,
        max_query_length: int = 128,
        **kwargs
    ):
        """
        Args:
            model_name_or_path: A T5 checkpoint name or path (e.g., 't5-large', 'google/t5-xxl-lm-adapt', etc.)
            verbose: Verbosity level.
            device: "cuda", "cpu", or None for auto.
            dtype: e.g. "float32", "float16", "bf16", or a torch.dtype.
            batch_size: How many documents to process at once.
            verbalizer_head: Prefixed to the doc text to mimic the 'Passage: ' from the original code.
            verbalizer: A short instruction appended to the doc text. The original UPR default is
                       "Please write a question based on this passage."
            max_input_length: Maximum tokens for the encoder side (document).
            max_query_length: Maximum tokens for the decoder side (query).
        """
        self.verbose = verbose
        self.device = get_device(device, self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size
        self.verbalizer_head = verbalizer_head
        self.verbalizer = verbalizer
        self.max_input_length = max_input_length
        self.max_query_length = max_query_length

        vprint(f"[UPR] Loading T5 model: {model_name_or_path}", self.verbose)
        vprint(f"[UPR] device={self.device}, dtype={self.dtype}, batch_size={batch_size}", self.verbose)

        # Load T5
        model_kwargs = kwargs.get("model_kwargs", {})
        self.model = T5ForConditionalGeneration.from_pretrained(
            model_name_or_path, torch_dtype=self.dtype, **model_kwargs
        ).to(self.device)
        self.model.eval()

        tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
        self.tokenizer = T5Tokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)

    def score(self, query: str, doc: str) -> float:
        """
        Score a single document. Negative log-likelihood of 'query' given 'doc'.
        Higher means more relevant (score = -NLL).
        """
        scores = self._get_scores(query, [doc])
        return scores[0] if scores else 0.0

    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        """
        Ranks a list of documents by the negative log-likelihood of the query given the doc.
        """
        # Convert user inputs into a list of Document objects
        docs = prep_docs(docs, doc_ids, metadata)

        # Score them
        doc_texts = [d.text for d in docs]
        scores = self._get_scores(query, doc_texts)

        # Sort in descending order of score
        ranked_results = [
            Result(document=doc, score=score, rank=idx + 1)
            for idx, (doc, score) in enumerate(
                sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    @torch.inference_mode()
    def _get_scores(self, query: str, docs: List[str]) -> List[float]:
        """
        Batched negative log-likelihood scoring:
            score = - sum_{tokens in query} [ log P(token | doc) ].
        """
        all_scores = []
        # Create mini-batches of docs
        for batch in get_chunks(docs, self.batch_size):
            # 1) Build the T5 encoder inputs for the doc
            #    (mimicking "Passage: {doc_text}. Please write a question..." from the original code)
            encoder_texts = [
                f"{self.verbalizer_head} {doc_text}. {self.verbalizer}"
                for doc_text in batch
            ]

            encoder_enc = self.tokenizer(
                encoder_texts,
                padding=True,
                truncation=True,
                max_length=self.max_input_length,
                return_tensors="pt",
            ).to(self.device)

            # 2) Build the T5 decoder labels for the query
            #    (the question is now the *label*, exactly as in original UPR).
            decoder_enc = self.tokenizer(
                [query] * len(batch),
                padding=True,
                truncation=True,
                max_length=self.max_query_length,
                return_tensors="pt",
            ).to(self.device)

            # 3) forward pass with `labels=...` so that T5 returns cross-entropy
            #    but we want the per-token log-likelihood to replicate the approach exactly.
            logits = self.model(
                input_ids=encoder_enc.input_ids,
                attention_mask=encoder_enc.attention_mask,
                labels=decoder_enc.input_ids,
            ).logits  # shape: [batch, seq_len, vocab_size]

            # 4) Compute log-softmax for each token
            log_probs = F.log_softmax(logits, dim=-1)  # [batch, seq_len, vocab_size]

            # 5) Gather the probabilities at each gold label token => negative log-likelihood
            #    next_token = decoder_enc.input_ids[..., 1:]
            #    but T5 shifts internally. We'll simply do gather on
            #    the label tokens and sum up, replicating the original.
            labels = decoder_enc.input_ids.unsqueeze(-1)  # [batch, seq_len, 1]
            token_log_probs = log_probs.gather(-1, labels).squeeze(-1)  # [batch, seq_len]

            # T5 shifts internally, so the first token is the "start token." The original UPR code
            # just sums everything. We'll do the same.
            nll = -token_log_probs  # [batch, seq_len]
            sum_nll = torch.sum(nll, dim=1)  # sum over query tokens

            # final score = - sum_nll (which is +ve if the NLL is large)
            # we want "best doc" to have the largest score => doc that yields the *lowest* NLL
            batch_scores = (-sum_nll).tolist()

            all_scores.extend(batch_scores)

        return all_scores
