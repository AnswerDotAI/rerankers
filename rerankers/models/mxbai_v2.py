"""
MXBai V2 Reranker implementation

Parts of the code borrowed/adapted from the Apache 2.0 licensed original codebase: https://github.com/mixedbread-ai/mxbai-rerank
"""

from __future__ import annotations

import torch
from typing import Union, List, Optional

# Rerankers base imports
from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from rerankers.results import RankedResults, Result
from rerankers.utils import vprint, prep_docs, get_device, get_dtype

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig


# Default prompt templates and tokens
SEPS = {
    "mixedbread-ai/mxbai-rerank-large-v2": "\n",
    "mixedbread-ai/mxbai-rerank-base-v2": "\n",
    "default": "\n"
}

INSTRUCTION_PROMPT = {
    "mixedbread-ai/mxbai-rerank-large-v2": "instruction: {instruction}",
    "mixedbread-ai/mxbai-rerank-base-v2": "instruction: {instruction}",
    "default": "instruction: {instruction}"
}

QUERY_PROMPT = {
    "mixedbread-ai/mxbai-rerank-large-v2": "query: {query}",
    "mixedbread-ai/mxbai-rerank-base-v2": "query: {query}",
    "default": "query: {query}"
}

DOC_PROMPT = {
    "mixedbread-ai/mxbai-rerank-large-v2": "document: {document}",
    "mixedbread-ai/mxbai-rerank-base-v2": "document: {document}",
    "default": "document: {document}"
}

TASK_PROMPT = {
    "mixedbread-ai/mxbai-rerank-large-v2": """You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).
Relevance:""",
    "mixedbread-ai/mxbai-rerank-base-v2": """You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).
Relevance:""",
    "default": """You are a search relevance expert who evaluates how well documents match search queries. For each query-document pair, carefully analyze the semantic relationship between them, then provide your binary relevance judgment (0 for not relevant, 1 for relevant).
Relevance:"""
}

CHAT_TEMPLATE = {
    "mixedbread-ai/mxbai-rerank-large-v2": {
        "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "mixedbread-ai/mxbai-rerank-base-v2": {
        "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    },
    "default": {
        "prefix": "<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n",
        "suffix": "<|im_end|>\n<|im_start|>assistant\n",
    }
}

POS_TOKEN = {
    "mixedbread-ai/mxbai-rerank-large-v2": "1",
    "mixedbread-ai/mxbai-rerank-base-v2": "1",
    "default": "1"
}

NEG_TOKEN = {
    "mixedbread-ai/mxbai-rerank-large-v2": "0",
    "mixedbread-ai/mxbai-rerank-base-v2": "0",
    "default": "0"
}


def _ensure_multiple_of_8(x: int, max_value: Optional[int] = None) -> int:
    """Make x a multiple of 8, respecting optional max_value"""
    if max_value is not None:
        max_value = max_value - max_value % 8
        x = min(x, max_value)
    return x - x % 8


class MxBaiV2Ranker(BaseRanker):
    """
    A reranker that uses MxBai models for yes/no-based relevance classification.
    
    This ranker uses causal language models from the MxBai family to determine
    document relevance by predicting binary relevance scores (0/1).
    """

    def __init__(
        self,
        model_name_or_path: str = "mixedbread-ai/mxbai-rerank-base-v2",
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[Union[str, torch.dtype]] = None,
        batch_size: int = 16,
        verbose: int = 1,
        max_length: int = 8192,
        **kwargs
    ):
        """
        Initialize the MxBai reranker.
        
        Args:
            model_name_or_path: Path or name of the MxBai model.
            device: Device to use (e.g. 'cpu', 'cuda:0', or 'auto').
            dtype: Torch dtype or 'auto'.
            batch_size: Batch size for processing multiple documents.
            verbose: Verbosity level.
            max_length: Maximum token length for inputs.
            **kwargs: Additional kwargs for model and tokenizer.
        """
        super().__init__(model_name_or_path=model_name_or_path, verbose=verbose)
        self.verbose = verbose
        self.device = get_device(device, verbose=self.verbose)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size
        self.max_length = max_length
        self.cfg = AutoConfig.from_pretrained(model_name_or_path)

        vprint(f"Loading MxBai model from {model_name_or_path}", self.verbose)
        vprint(f"Device: {self.device}, Dtype: {self.dtype}", self.verbose)

        # Extract model kwargs
        tokenizer_kwargs = kwargs.get("tokenizer_kwargs", {})
        model_kwargs = kwargs.get("model_kwargs", {})

        # Try to use flash attention if available
        try:
            import flash_attn
            attn_impl = "flash_attention_2"
        except ImportError: 
            attn_impl = None

        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            attn_implementation=attn_impl,
            torch_dtype=self.dtype if isinstance(self.dtype, torch.dtype) else "auto",
            device_map=str(self.device),
            **model_kwargs,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, **tokenizer_kwargs)
        self.tokenizer.padding_side = "left"

        # Get model-specific templates and tokens
        model_key = model_name_or_path.split("/")[-1]
        self._setup_templates(model_key)
        
        # Pre-tokenize static prompts for efficiency
        self._prepare_tokenized_templates()
        
        # Switch to eval mode
        self.model.eval()
        vprint("MxBaiV2Ranker ready.", self.verbose)

    def _setup_templates(self, model_key: str):
        """Set up the templates and tokens for the specific model"""
        # Helper function to get template with fallback to default
        def get_template(template_dict, key):
            return template_dict.get(key, template_dict["default"])
        
        # Set up all templates
        self.task_prompt = get_template(TASK_PROMPT, model_key)
        self.chat_template = get_template(CHAT_TEMPLATE, model_key)
        self.query_prompt = get_template(QUERY_PROMPT, model_key)
        self.doc_prompt = get_template(DOC_PROMPT, model_key)
        self.instruction_prompt = get_template(INSTRUCTION_PROMPT, model_key)
        self.pos_token = get_template(POS_TOKEN, model_key)
        self.neg_token = get_template(NEG_TOKEN, model_key)
        self.sep = get_template(SEPS, model_key)
        
        if not any(model_key in template_dict for template_dict in [TASK_PROMPT, CHAT_TEMPLATE, QUERY_PROMPT, DOC_PROMPT, INSTRUCTION_PROMPT, POS_TOKEN, NEG_TOKEN, SEPS]):
            vprint("Model name did not have all necessary instructions. Using default prompt formats which might not be suitable for this model!", self.verbose)

    def _prepare_tokenized_templates(self):
        """Pre-tokenize templates for efficiency"""
        # Get token IDs for positive and negative tokens
        self.pos_id = self.tokenizer(self.pos_token, return_tensors=None, add_special_tokens=False)["input_ids"][0]
        self.neg_id = self.tokenizer(self.neg_token, return_tensors=None, add_special_tokens=False)["input_ids"][0]
        
        # Pre-tokenize chat template parts
        self.prefix_ids = self.tokenizer(self.chat_template["prefix"], return_tensors=None, add_special_tokens=False)["input_ids"]
        self.suffix_ids = self.tokenizer(self.chat_template["suffix"], return_tensors=None, add_special_tokens=False)["input_ids"]
        
        # Pre-tokenize task prompt and separator
        self.task_prompt_ids = self.tokenizer(self.task_prompt, return_tensors=None, add_special_tokens=False)["input_ids"]
        self.sep_ids = self.tokenizer(self.sep, return_tensors=None, add_special_tokens=False)["input_ids"]
        
        # Calculate total length of static tokens
        self.static_tokens_length = (
            len(self.prefix_ids) +
            len(self.task_prompt_ids) +
            len(self.suffix_ids) +
            len(self.sep_ids)
        )
        
        # Set model max length
        self.model_max_length = self.cfg.max_position_embeddings
        
        # Adjust max_length to account for static tokens
        if self.max_length + self.static_tokens_length > self.model_max_length:
            self.max_length = self.model_max_length - self.static_tokens_length
        
        # Ensure padding length is a multiple of 8 for efficiency
        self.padding_length = _ensure_multiple_of_8(
            max(self.model_max_length, self.max_length + self.static_tokens_length), 
            max_value=self.model_max_length
        )

    def _create_full_input_ids(self, content_ids: List[int]) -> List[int]:
        """
        Create the full input by combining content with template parts.
        
        Args:
            content_ids: Token IDs for the query-document content
            
        Returns:
            List of token IDs for the complete input
        """
        return (
            self.prefix_ids +
            content_ids +
            self.sep_ids +
            self.task_prompt_ids +
            self.suffix_ids
        )

    def _prepare_batch(
        self,
        queries: List[str],
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> dict:
        """
        Prepare a batch of query-document pairs for the model.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            instruction: Optional instruction to prepend
            
        Returns:
            Dictionary with input_ids and attention_mask tensors
        """
        batch_inputs = []
        
        for query, document in zip(queries, documents):
            # Format query with template
            query_text = self.query_prompt.format(query=query)
            
            # Add instruction if provided
            if instruction:
                instruction_text = self.instruction_prompt.format(instruction=instruction)
                query_text = instruction_text + self.sep + query_text
            
            # Tokenize query with length limit
            query_ids = self.tokenizer(
                query_text,
                return_tensors=None,
                add_special_tokens=False,
                max_length=self.max_length * 3 // 4,  # Use 3/4 of tokens for query
                truncation=True,
            )["input_ids"]
            
            # Calculate remaining tokens for document
            available_tokens = self.model_max_length - len(query_ids) - self.static_tokens_length
            doc_max_length = min(available_tokens, self.max_length // 4)  # Use 1/4 of tokens for document
            
            # Tokenize document
            doc_text = self.doc_prompt.format(document=document)
            doc_ids = self.tokenizer(
                doc_text,
                return_tensors=None,
                add_special_tokens=False,
                max_length=doc_max_length,
                truncation=True,
            )["input_ids"]
            
            # Combine query and document
            combined = self.tokenizer.prepare_for_model(
                query_ids,
                self.sep_ids + doc_ids,
                truncation="only_second",
                max_length=self.max_length,
                padding=False,
                return_attention_mask=False,
                return_token_type_ids=False,
                add_special_tokens=False,
            )
            
            # Create full input with template
            full_input_ids = self._create_full_input_ids(combined["input_ids"])
            
            # Add to batch
            batch_inputs.append({
                "input_ids": full_input_ids,
                "attention_mask": [1] * len(full_input_ids),
            })
        
        # Pad all inputs to the same length
        padded_batch = self.tokenizer.pad(
            batch_inputs,
            padding="longest",
            max_length=self.padding_length,
            pad_to_multiple_of=8,
            return_tensors="pt",
        )
        
        return padded_batch

    @torch.inference_mode()
    def _predict(
        self,
        queries: List[str],
        documents: List[str],
        instruction: Optional[str] = None,
    ) -> torch.Tensor:
        """
        Get relevance scores for query-document pairs.
        
        Args:
            queries: List of query strings
            documents: List of document strings
            instruction: Optional instruction to prepend
            
        Returns:
            Tensor of relevance scores
        """
        # Prepare inputs
        inputs = self._prepare_batch(queries, documents, instruction=instruction)
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Run model
        outputs = self.model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            output_hidden_states=True,
        )
        
        score = outputs.logits[:, -1, self.pos_id] - outputs.logits[:, -1, self.neg_id]
        
        # Return scores as CPU tensor
        return score.detach().cpu().float()

    @torch.inference_mode()
    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
        batch_size: Optional[int] = None,
        instruction: Optional[str] = None,
    ) -> RankedResults:
        """
        Rank documents by relevance to the query.
        
        Args:
            query: Query string
            docs: Documents to rank
            doc_ids: Optional document IDs
            metadata: Optional document metadata
            batch_size: Optional batch size override
            instruction: Optional instruction to prepend
            
        Returns:
            RankedResults with documents sorted by relevance
        """
        # Prepare documents
        docs = prep_docs(docs, doc_ids, metadata)
        
        # Use default batch size if not specified
        if batch_size is None:
            batch_size = self.batch_size
        
        all_docs = []
        all_scores = []
        
        # Process in batches
        for i in range(0, len(docs), batch_size):
            batch_docs = docs[i:i + batch_size]
            batch_scores = self._predict(
                queries=[query] * len(batch_docs),
                documents=[d.text for d in batch_docs],
                instruction=instruction,
            )
            
            all_docs.extend(batch_docs)
            all_scores.extend(batch_scores.tolist())
        
        # Sort by descending score
        scored_docs = sorted(zip(all_docs, all_scores), key=lambda x: x[1], reverse=True)
        
        # Create ranked results
        results = [
            Result(document=doc, score=score, rank=idx + 1)
            for idx, (doc, score) in enumerate(scored_docs)
        ]
        
        return RankedResults(results=results, query=query, has_scores=True)

    @torch.inference_mode()
    def score(
        self,
        query: str,
        doc: Union[str, Document],
        instruction: Optional[str] = None,
    ) -> float:
        """
        Score a single query-document pair.
        
        Args:
            query: Query string
            doc: Document to score
            instruction: Optional instruction to prepend
            
        Returns:
            Relevance score as float
        """
        # Extract text if document is a Document object
        doc_text = doc.text if isinstance(doc, Document) else doc
        
        # Get score
        scores = self._predict(
            queries=[query],
            documents=[doc_text],
            instruction=instruction,
        )
        
        # Return as float
        return float(scores[0])