import math
from typing import List, Optional, Union

import torch
from transformers import pipeline

from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result
from rerankers.documents import Document
from rerankers.utils import prep_docs


class EcoRank(BaseRanker):
    def __init__(self, model_name_or_path: str, verbose: int = 1, **kwargs):
        """
        EcoRank implements the two-stage ranking method from the EcoRank paper.
        It uses two text2text-generation pipelines (a “cheap” and an “expensive” model)
        to first filter passages (binary stage) and then re-rank them (pairwise stage).

        Keyword Args (with defaults):
            cheap_modelcard: model card for the cheap model (default "google/flan-t5-large")
            exp_modelcard: model card for the expensive model (default "google/flan-t5-xl")
            budget_tokens: overall token budget (default 4000)
            budget_split_x: fraction of token budget for stage 1 (default 0.5)
            budget_split_y: fraction of token budget for stage 2 (default 0.5)
            binary_prompt_head_len: constant token cost for binary prompt (default 15)
            binary_output_possible_len: constant token cost for binary output (default 1)
            prp_prompt_head_len: constant token cost for pairwise prompt (default 25)
            prp_output_possible_len: constant token cost for pairwise output (default 2)
            total_passages: number of passages to consider (default 50)
        """
        self.verbose = verbose
        self.cheap_modelcard = kwargs.get("cheap_modelcard", "google/flan-t5-large")
        self.exp_modelcard = kwargs.get("exp_modelcard", "google/flan-t5-xl")
        self.budget_tokens = kwargs.get("budget_tokens", 4000)
        self.budget_split_x = kwargs.get("budget_split_x", 0.5)
        self.budget_split_y = kwargs.get("budget_split_y", 0.5)
        self.binary_prompt_head_len = kwargs.get("binary_prompt_head_len", 15)
        self.binary_output_possible_len = kwargs.get("binary_output_possible_len", 1)
        self.prp_prompt_head_len = kwargs.get("prp_prompt_head_len", 25)
        self.prp_output_possible_len = kwargs.get("prp_output_possible_len", 2)
        self.total_passages = kwargs.get("total_passages", 50)

        # Determine device: if CUDA is available, use device 0; else use CPU (device=-1)
        device = 0 if torch.cuda.is_available() else -1
        if self.verbose:
            print(f"Loading EcoRank models on device: {device}")

        self.cheap_model = pipeline("text2text-generation", model=self.cheap_modelcard, device=device)
        self.exp_model = pipeline("text2text-generation", model=self.exp_modelcard, device=device)

    def get_binary_response(self, passage: str, query: str, model_size: str) -> str:
        prompt = (
            f"Is the following passage related to the query?\n"
            f"passage: {passage}\n"
            f"query: {query}\n"
            "Answer in yes or no"
        )
        if model_size == 'expensive':
            ans = self.exp_model(prompt)[0]['generated_text']
        else:
            ans = self.cheap_model(prompt)[0]['generated_text']
        return ans

    def get_prp_response(self, query: str, passage1: str, passage2: str, model_size: str) -> str:
        prompt = f"""Given a query "{query}", which of the following two passages is more relevant to the query?
Passage A: {passage1}
Passage B: {passage2}
Output Passage A or Passage B.
"""
        if model_size == 'expensive':
            ans = self.exp_model(prompt)[0]['generated_text']
        else:
            ans = self.cheap_model(prompt)[0]['generated_text']
        return ans

    def count_top_l(self, token_limit: int, query_len: int, docs: List[Document]) -> int:
        tokens = 0
        howmany = 0
        docs_to_consider = docs[:self.total_passages]
        for idx in range(len(docs_to_consider) - 1):
            text1 = docs_to_consider[idx].text
            text2 = docs_to_consider[idx + 1].text
            text1_len = len(text1.split())
            text2_len = len(text2.split())
            possible_tokens = self.prp_prompt_head_len + self.prp_output_possible_len + query_len + text1_len + text2_len
            if tokens + possible_tokens < token_limit:
                tokens += possible_tokens
                howmany += 1
            else:
                break
        return howmany

    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
        k: Optional[int] = None,
    ) -> RankedResults:
        """
        Ranks documents based on relevance to the query.
        
        Args:
            query (str): The query string.
            docs (Union[str, List[str], Document, List[Document]]): One or more documents.
            doc_ids (Optional[Union[List[str], List[int]]]): Document IDs.
            metadata (Optional[List[dict]]): Additional metadata.
            k (Optional[int]): Maximum number of results to return.
        
        Returns:
            RankedResults: The ranked documents.
        """
        # Normalize docs using the common helper (this sets doc_id if provided)
        docs = prep_docs(docs, doc_ids, metadata)

        # Stage 1: Binary classification to filter passages under a simulated token budget.
        BINARY_TOKEN_LIMIT = int(self.budget_split_x * self.budget_tokens)
        binary_running_token = 0
        yes_docs = []
        no_docs = []
        token_reached = False
        ending_idx = 0
        query_len = len(query.split())

        for idx, doc in enumerate(docs[:self.total_passages]):
            passage = doc.text
            passage_len = len(passage.split())
            token_cost = self.binary_prompt_head_len + passage_len + query_len + self.binary_output_possible_len
            if binary_running_token + token_cost < BINARY_TOKEN_LIMIT:
                binary_running_token += token_cost
                try:
                    ans = self.get_binary_response(passage, query, 'expensive').strip().lower()
                except Exception as e:
                    ans = ""
                if "yes" in ans:
                    yes_docs.append(doc)
                elif "no" in ans:
                    no_docs.append(doc)
                else:
                    # In case of ambiguity, default to positive.
                    yes_docs.append(doc)
            else:
                token_reached = True
                ending_idx = idx
                break

        # Build the initial ordering:
        reranked_docs = yes_docs.copy()
        if token_reached:
            reranked_docs.extend(docs[ending_idx:self.total_passages])
        reranked_docs.extend(no_docs)

        # Stage 2: Pairwise re-ranking via bubble-sort–like passes.
        PRP_TOKEN_LIMIT = int(self.budget_split_y * self.budget_tokens) * 3
        top_l = self.count_top_l(PRP_TOKEN_LIMIT, query_len, reranked_docs)
        full_value = math.ceil(top_l / self.total_passages) if self.total_passages > 0 else 0
        cntr = full_value
        mod_value = top_l % self.total_passages
        while cntr > 0:
            limit_val = self.total_passages if cntr > 1 else (mod_value if mod_value else self.total_passages)
            for idx in range(limit_val - 1, 0, -1):
                passage_b = reranked_docs[idx].text
                passage_a = reranked_docs[idx - 1].text
                try:
                    ans = self.get_prp_response(query, passage_b, passage_a, 'cheap').strip().lower()
                except Exception as e:
                    ans = ""
                if ans == "passage a":
                    # Swap documents so that the more relevant passage comes first.
                    reranked_docs[idx], reranked_docs[idx - 1] = reranked_docs[idx - 1], reranked_docs[idx]
            cntr -= 1

        # Build final results: if no doc was a "yes" in stage 1, set score = 0 for all.
        results = []
        if yes_docs:
            base_score = 1.0
            decrement = 0.001
        else:
            base_score = 0.0
            decrement = 0.0

        for i, doc in enumerate(reranked_docs):
            if doc.doc_id is None:
                doc.doc_id = f"doc_{i+1}"
            score = base_score - (i * decrement)
            results.append(Result(document=doc, rank=i + 1, score=score))

        # Truncate the final list to k results if k is provided.
        if k is not None:
            results = results[:k]

        return RankedResults(results=results, query=query, has_scores=True)

    def score(self, query: str, doc: str) -> float:
        """
        For a single document, we run the binary stage and return 1.0 if the answer contains "yes",
        and 0.0 otherwise.
        """
        try:
            ans = self.get_binary_response(doc, query, 'cheap').strip().lower()
            return 1.0 if "yes" in ans else 0.0
        except Exception as e:
            return 0.0
