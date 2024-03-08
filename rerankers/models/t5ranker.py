"""
Code for InRanker is taken from the excellent InRanker repo https://github.com/unicamp-dl/InRanker under its Apache 2.0 license.
The only change to the original implementation is the removal of InRanker's BaseRanker, replacing it with our own to support the unified API better.
The main purpose for adapting this code here rather than installing the InRanker library is to ensure greater version compatibility (InRanker requires Python >=3.10)
"""

from typing import List, Optional, Union
from math import ceil

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from rerankers.models.ranker import BaseRanker

import torch

from rerankers.results import RankedResults, Result
from rerankers.utils import (
    vprint,
    get_device,
    get_dtype,
    ensure_docids,
    ensure_docs_list,
    get_chunks,
)

PREDICTION_TOKENS = {
    "default": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco": ["▁false", "▁true"],
    "castorini/monot5-base-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco": ["▁false", "▁true"],
    "castorini/monot5-large-msmarco-10k": ["▁false", "▁true"],
    "castorini/monot5-base-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-med-msmarco": ["▁false", "▁true"],
    "castorini/monot5-3b-msmarco-10k": ["▁false", "▁true"],
    "unicamp-dl/InRanker-small": ["▁false", "▁true"],
    "unicamp-dl/InRanker-base": ["▁false", "▁true"],
    "unicamp-dl/InRanker-3B": ["▁false", "▁true"],
    "unicamp-dl/mt5-base-en-msmarco": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-100k-v2": ["▁não", "▁sim"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v2": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-en-pt-msmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/mt5-base-mmarco-v1": ["▁no", "▁yes"],
    "unicamp-dl/ptt5-base-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-pt-msmarco-100k-v1": ["▁não", "▁sim"],
    "unicamp-dl/ptt5-base-en-pt-msmarco-10k-v1": ["▁não", "▁sim"],
    "unicamp-dl/mt5-3B-mmarco-en-pt": ["▁", "▁true"],
    "unicamp-dl/mt5-13b-mmarco-100k": ["▁", "▁true"],
}


def _get_output_tokens(model_name_or_path, token_false: str, token_true: str):
    if token_false == "auto":
        if model_name_or_path in PREDICTION_TOKENS:
            token_false = PREDICTION_TOKENS[model_name_or_path][0]
        else:
            print(
                f"WARNING: Model {model_name_or_path} does not have known True/False tokens. Defaulting token_false to `{token_false}`."
            )
    if token_true == "auto":
        if model_name_or_path in PREDICTION_TOKENS:
            token_true = PREDICTION_TOKENS[model_name_or_path][1]
        else:
            token_true = PREDICTION_TOKENS["default"][1]
            print(
                f"WARNING: Model {model_name_or_path} does not have known True/False tokens. Defaulting token_true to `{token_true}`."
            )

    return token_false, token_true


class T5Ranker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        batch_size: int = 32,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        verbose: int = 1,
        token_false: str = "auto",
        token_true: str = "auto",
        return_logits: bool = False,
    ):
        """
        Implementation of the key functions from https://github.com/unicamp-dl/InRanker/blob/main/inranker/rankers.py
        Changes are detailed in the docstring for each relevant function.

        T5Ranker is a wrapper for using Seq2Seq models for ranking.
        Args:
            batch_size: The batch size to use when encoding.
            dtype: Data type for model weights.
            device: The device to use for inference ("cpu", "cuda", or "mps").
            verbose: Verbosity level.
            silent: Whether to show progress bars.
        """
        self.verbose = verbose
        self.device = get_device(device, self.verbose, no_mps=True)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size
        vprint(
            f"Loading model {model_name_or_path}, this might take a while...",
            self.verbose,
        )
        vprint(f"Using device {self.device}.", self.verbose)
        vprint(f"Using dtype {self.dtype}.", self.verbose)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        token_false, token_true = _get_output_tokens(
            model_name_or_path=model_name_or_path,
            token_false=token_false,
            token_true=token_true,
        )
        self.token_false_id = self.tokenizer.convert_tokens_to_ids(token_false)
        self.token_true_id = self.tokenizer.convert_tokens_to_ids(token_true)
        vprint(f"T5 true token set to {token_true}", self.verbose)
        vprint(f"T5 false token set to {token_false}", self.verbose)

        self.return_logits = return_logits
        if self.return_logits:
            vprint(
                f"Returning raw logits for `{token_true}` as scores...", self.verbose
            )
        else:
            vprint("Returning normalised scores...", self.verbose)

    def rank(
        self,
        query: str,
        docs: List[str],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
    ) -> RankedResults:
        """
        Ranks a list of documents based on their relevance to the query.
        """
        docs = ensure_docs_list(docs)
        doc_ids = ensure_docids(doc_ids, len(docs))
        scores = self._get_scores(query, docs)
        ranked_results = [
            Result(doc_id=doc_id, text=doc, score=score, rank=idx + 1)
            for idx, (doc_id, doc, score) in enumerate(
                sorted(zip(doc_ids, docs, scores), key=lambda x: x[2], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def score(self, query: str, doc: str) -> float:
        """
        Scores a single document's relevance to a query.
        """
        scores = self._get_scores(query, [doc])
        return scores[0] if scores else 0.0

    @torch.no_grad()
    def _get_scores(
        self,
        query: str,
        docs: List[str],
        max_length: int = 512,
        batch_size: Optional[int] = None,
    ) -> List[float]:
        """
        Implementation from https://github.com/unicamp-dl/InRanker/blob/main/inranker/rankers.py.
        Lightly modified so only the positive logits are returned and renamed the chunking function.

        Given a query and a list of documents, return a list of scores.
        Args:
            query: The query string.
            docs: A list of document strings.
            max_length: The maximum length of the input sequence.
        """
        if self.return_logits:
            logits = []
        else:
            scores = []
        if batch_size is None:
            batch_size = self.batch_size
        for batch in tqdm(
            get_chunks(docs, batch_size),
            disable=not self.verbose,
            desc="Scoring...",
            total=ceil(len(docs) / batch_size),
        ):
            queries_documents = [
                f"Query: {query} Document: {text} Relevant:" for text in batch
            ]
            tokenized = self.tokenizer(
                queries_documents,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=max_length,
            ).to(self.device)
            input_ids = tokenized["input_ids"]
            attention_mask = tokenized["attention_mask"]

            _, batch_scores = self._greedy_decode(
                model=self.model,
                input_ids=input_ids,
                length=1,
                attention_mask=attention_mask,
                return_last_logits=True,
            )
            batch_scores = batch_scores[
                :, [self.token_false_id, self.token_true_id]
            ].cpu()
            if self.return_logits:
                logits.extend(batch_scores[:, 1].tolist())
            else:
                batch_scores = torch.log_softmax(batch_scores, dim=-1)
                batch_scores = torch.exp(batch_scores[:, 1])
                scores.extend(batch_scores.tolist())

        if self.return_logits:
            return logits
        return scores

    @torch.no_grad()
    def _greedy_decode(
        self,
        model,
        input_ids: torch.Tensor,
        length: int,
        attention_mask: torch.Tensor = None,
        return_last_logits: bool = True,
    ):
        """Implementation from https://github.com/unicamp-dl/InRanker/blob/main/inranker/rankers.py"""
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        next_token_logits = None
        for _ in range(length):
            model_inputs = model.prepare_inputs_for_generation(
                decode_ids,
                encoder_outputs=encoder_outputs,
                past=None,
                attention_mask=attention_mask,
                use_cache=True,
            )
            outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
            decode_ids = torch.cat(
                [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
            )
        if return_last_logits:
            return decode_ids, next_token_logits
        return decode_ids
