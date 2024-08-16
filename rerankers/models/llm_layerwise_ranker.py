import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from typing import Union, List, Optional
from rerankers.utils import vprint, get_device, get_dtype, prep_docs
from rerankers.results import RankedResults, Result


PROMPTS = {
    "BAAI/bge-reranker-v2.5-gemma2-lightweight": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.",
    "default": "Given a query A and a passage B, determine whether the passage contains an answer to the query by providing a prediction of either 'Yes' or 'No'.",
}

DEFAULT_PARAMS = {
    "default": {},
    "BAAI/bge-multilingual-gemma2": {},
    "BAAI/bge-reranker-v2-gemma": {},
    "BAAI/bge-reranker-v2-minicpm-layerwise": {"cutoff_layers": [28]},
    "BAAI/bge-reranker-v2.5-gemma2-lightweight": {
        "cutoff_layers": [28],
        "compress_ratio": 2,
        "compress_layer": [24, 40],
    },
}


class LLMLayerWiseRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str = "BAAI/bge-reranker-v2.5-gemma2-lightweight",
        max_sequence_length: int = 512,
        dtype: Optional[Union[str, torch.dtype]] = None,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 16,
        verbose: int = 1,
        prompt: Optional[str] = None,
        cutoff_layers: Optional[List[int]] = None,
        compress_ratio: Optional[int] = None,
        compress_layer: Optional[List[int]] = None,
    ):
        self.verbose = verbose
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size

        vprint(
            f"Loading model {model_name_or_path}, this might take a while...",
            self.verbose,
        )
        vprint(f"Using device {self.device}.", self.verbose)
        vprint(f"Using dtype {self.dtype}.", self.verbose)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        self.max_sequence_length = max_sequence_length
        self.tokenizer.model_max_length = self.max_sequence_length
        self.tokenizer.padding_side = "right"

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, trust_remote_code=True, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

        # Create params dict based on specified values or defaults
        params = {}
        if cutoff_layers is not None:
            params["cutoff_layers"] = cutoff_layers
        if compress_ratio is not None:
            params["compress_ratio"] = compress_ratio
        if compress_layer is not None:
            params["compress_layer"] = compress_layer
        if not params:
            params = DEFAULT_PARAMS.get(model_name_or_path, DEFAULT_PARAMS["default"])
        self.params = params

        self.prompt = prompt
        if self.prompt is None:
            self.prompt = PROMPTS.get(model_name_or_path, PROMPTS["default"])

    def _get_inputs(self, pairs, max_sequence_length: int):
        prompt = self.prompt
        sep = "\n"
        prompt_inputs = self.tokenizer(
            prompt, return_tensors=None, add_special_tokens=False
        )["input_ids"]
        sep_inputs = self.tokenizer(sep, return_tensors=None, add_special_tokens=False)[
            "input_ids"
        ]
        inputs = []
        for query, passage in pairs:
            query_inputs = self.tokenizer(
                f"A: {query}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_sequence_length * 3 // 4,
                truncation=True,
            )
            passage_inputs = self.tokenizer(
                f"B: {passage}",
                return_tensors=None,
                add_special_tokens=False,
                max_length=max_sequence_length,
                truncation=True,
            )
            item = self.tokenizer.prepare_for_model(
                [self.tokenizer.bos_token_id] + query_inputs["input_ids"],
                sep_inputs + passage_inputs["input_ids"],
                truncation="only_second",
                max_length=max_sequence_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            item["input_ids"] = item["input_ids"] + sep_inputs + prompt_inputs
            item["attention_mask"] = [1] * len(item["input_ids"])
            inputs.append(item)

        return self.tokenizer.pad(
            inputs,
            padding=True,
            max_length=max_sequence_length + len(sep_inputs) + len(prompt_inputs),
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

    @torch.no_grad()
    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)
        pairs = [(query, doc.text) for doc in docs]

        # Override self.batch_size if explicitly set
        if batch_size is None:
            batch_size = self.batch_size

        # Same for max_sequence_length
        if max_sequence_length is None:
            max_sequence_length = self.max_sequence_length

        batched_pairs = [
            pairs[i : i + batch_size] for i in range(0, len(pairs), batch_size)
        ]
        scores = []

        for batch in batched_pairs:
            inputs = self._get_inputs(batch, max_sequence_length=max_sequence_length)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            outputs = self.model(**inputs, return_dict=True, **self.params)
            all_scores = [
                scores[:, -1]
                .view(
                    -1,
                )
                .float()
                for scores in outputs[0]
            ]
            batch_scores = all_scores[-1].cpu().numpy().tolist()

            scores.extend(batch_scores)

        ranked_results = [
            Result(document=doc, score=score, rank=idx + 1)
            for idx, (doc, score) in enumerate(
                sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    @torch.no_grad()
    def score(self, query: str, doc: str) -> float:
        inputs = self._get_inputs(
            [(query, doc)], max_sequence_length=self.max_sequence_length
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs, return_dict=True, **self.params)
        all_scores = [
            scores[:, -1]
            .view(
                -1,
            )
            .float()
            for scores in outputs[0]
        ]
        score = all_scores[-1].item()

        return score
