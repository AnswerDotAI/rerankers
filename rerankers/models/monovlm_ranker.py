import torch
from PIL import Image
import base64
import io
# TODO: Support more than Qwen
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from typing import Union, List, Optional
from rerankers.utils import vprint, get_device, get_dtype, prep_image_docs
from rerankers.results import RankedResults, Result

PREDICTION_TOKENS = {
    "default": ["False", "True"],
    "lightonai/MonoQwen2-VL-v0.1": ["False", "True"]
}

def _get_output_tokens(model_name_or_path, token_false: str, token_true: str):
    if token_false == "auto":
        if model_name_or_path in PREDICTION_TOKENS:
            token_false = PREDICTION_TOKENS[model_name_or_path][0]
        else:
            token_false = PREDICTION_TOKENS["default"][0]
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

class MonoVLMRanker(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        processor_name: Optional[str] = None,
        dtype: Optional[Union[str, torch.dtype]] = 'bf16',
        device: Optional[Union[str, torch.device]] = None,
        batch_size: int = 1,
        verbose: int = 1,
        token_false: str = "auto",
        token_true: str = "auto",
        return_logits: bool = False,
        prompt_template: str = "Assert the relevance of the previous image document to the following query, answer True or False. The query is: {query}",
        **kwargs
    ):
        self.verbose = verbose
        self.device = get_device(device, verbose=self.verbose)
        if self.device == 'mps':
            print("WARNING: MPS is not supported by MonoVLMRanker due to PyTorch limitations. Falling back to CPU.")
            self.device = 'cpu'
        print(dtype)
        self.dtype = get_dtype(dtype, self.device, self.verbose)
        self.batch_size = batch_size
        self.return_logits = return_logits
        self.prompt_template = prompt_template

        vprint(f"Loading model {model_name_or_path}, this might take a while...", self.verbose)
        vprint(f"Using device {self.device}.", self.verbose)
        vprint(f"Using dtype {self.dtype}.", self.verbose)

        processor_name = processor_name or "Qwen/Qwen2-VL-2B-Instruct"
        processor_kwargs = kwargs.get("processor_kwargs", {})
        model_kwargs = kwargs.get("model_kwargs", {})
        attention_implementation = kwargs.get("attention_implementation", "flash_attention_2")
        self.processor = AutoProcessor.from_pretrained(processor_name, **processor_kwargs)
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_name_or_path,
            device_map=self.device,
            torch_dtype=self.dtype,
            attn_implementation=attention_implementation,
            **model_kwargs
        )
        self.model.eval()

        token_false, token_true = _get_output_tokens(
            model_name_or_path=model_name_or_path,
            token_false=token_false,
            token_true=token_true,
        )
        self.token_false_id = self.processor.tokenizer.convert_tokens_to_ids(token_false)
        self.token_true_id = self.processor.tokenizer.convert_tokens_to_ids(token_true)
        
        vprint(f"VLM true token set to {token_true}", self.verbose)
        vprint(f"VLM false token set to {token_false}", self.verbose)

    @torch.inference_mode()
    def _get_scores(self, query: str, docs: List[Document]) -> List[float]:
        scores = []
        for doc in docs:
            if doc.document_type != "image" or not doc.base64:
                raise ValueError("MonoVLMRanker requires image documents with base64 data")
            
            # Convert base64 to PIL Image
            image_io = io.BytesIO(base64.b64decode(doc.base64))
            image_io.seek(0)  # Reset file pointer to start
            image = Image.open(image_io).convert('RGB')

            # Prepare prompt
            prompt = self.prompt_template.format(query=query)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Process inputs
            text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            inputs = self.processor(
                text=text, 
                images=image, 
                return_tensors="pt"
            ).to(self.device).to(self.dtype)

            # Get model outputs
            outputs = self.model(**inputs)
            logits = outputs.logits[:, -1, :]
            
            # Calculate scores
            relevant_logits = logits[:, [self.token_false_id, self.token_true_id]]
            if self.return_logits:
                score = relevant_logits[0, 1].cpu().item()  # True logit
            else:
                probs = torch.softmax(relevant_logits, dim=-1)
                score = probs[0, 1].cpu().item()  # True probability
            
            scores.append(score)
            
        return scores

    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        docs = prep_image_docs(docs, doc_ids, metadata)
        scores = self._get_scores(query, docs)
        ranked_results = [
            Result(document=doc, score=score, rank=idx + 1)
            for idx, (doc, score) in enumerate(
                sorted(zip(docs, scores), key=lambda x: x[1], reverse=True)
            )
        ]
        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def score(self, query: str, doc: Union[str, Document]) -> float:
        scores = self._get_scores(query, [doc])
        return scores[0]
