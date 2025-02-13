from typing import List, Optional, Union, Dict
import warnings
import re

try:
    from litellm import completion
except ImportError:
    pass

from rerankers.models.ranker import BaseRanker
from rerankers.documents import Document
from rerankers.results import RankedResults, Result
from rerankers.utils import prep_docs, vprint

SUPPORTED_BACKENDS = ["litellm"]
UNSUPPORTED_BACKENDS = ["gaspard", "claudette", "cosette"]

SYSTEM = (
    "You are a friendly AI assistant, working on document relevance filtering. Your task is "
    "to determine if a document is relevant to answering a given query. You must assign a binary "
    "RELEVANT or NOT_RELEVANT label to each document by carefully analysing them and the query."
)
DEFAULT_PROMPT_TEMPLATE = """<instructions>
Think carefully about whether the following documents would be useful to answer the query.
For each document, explain your reasoning and then provide a binary decision (RELEVANT or NOT_RELEVANT). If a document is partially relevant, you will assign the RELEVANT label.

The documents will be given to you in the following format:

<input>
<query>
Text of the query.
</query>

<documents>
<document id=0>
Text of the first document.
</document>
<document id=1>
Text of the second document.
</document>
</documents>
</input>
And you will respond in the following format:

<document id=X>
<explanation>
Your reasoning regarding the document's relevance.
</explanation>
<answer>
RELEVANT or NOT_RELEVANT
</answer>
</document>
</instructions>

Here is the query and documents:

<input>
<query>
{query}
</query>

<documents>
{docu_inputs}
</documents>
</input>

Analyse the above documents and provide your responses using the provided format. You must assign either the RELEVANT or NOT_RELEVANT label, no other option is permitted."""

class LLMRelevanceFilter(BaseRanker):
    def __init__(
        self,
        model_name_or_path: str,
        backend: str = "litellm",
        prompt_template: Optional[str] = None,
        temperature: float = 0.0,
        verbose: int = 1,
        default_label: str = "RELEVANT",
        **kwargs
    ):
        """Initialize the LLM Relevance Filter.
        
        Args:
            model_name_or_path: Name of the model to use (e.g. "gpt-4")
            backend: One of "litellm", "gaspard", "claudette", "cosette"
            prompt_template: Optional custom prompt template. Must include {query} and {docu_inputs} placeholders.
            temperature: Temperature for LLM sampling (default 0.0 for deterministic outputs)
            verbose: Verbosity level
            **kwargs: Additional kwargs passed to the backend
        """
        super().__init__(model_name_or_path, verbose)
        if backend not in SUPPORTED_BACKENDS:
            raise ValueError(f"Backend must be one of {SUPPORTED_BACKENDS}")
        
        if backend != "litellm":
            warnings.warn(f"Backend {backend} is experimental and may not work as expected")
            
        self.backend = backend
        self.model_name = model_name_or_path
        self.temperature = temperature
        self.prompt_template = prompt_template or DEFAULT_PROMPT_TEMPLATE
        self.verbose = verbose
        self.additional_kwargs = kwargs
        self.default_label = default_label
        
        vprint(f"Initialized LLMRelevanceFilter with {backend} backend using model {model_name_or_path}", verbose)
        
        # Verify backend is available
        if backend == "litellm" and "completion" not in globals():
            raise ImportError("litellm is required for the litellm backend. Install with pip install litellm")
    
    def _get_completion(self, prompt: str) -> str:
        """Get completion from the selected backend."""
        if self.backend == "litellm":
            response = completion(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                **self.additional_kwargs
            )
            return response.choices[0].message.content.strip()
        else:
            raise NotImplementedError(f"Backend {self.backend} not yet implemented")
    
    def _parse_response(self, response: str) -> str:
        """
        Parse an XML response to extract the answer from within the <answer> tags.
        If no answer is found, defaults to "NOT_RELEVANT".
        """
        match = re.search(r'<answer>\s*(RELEVANT|NOT_RELEVANT)\s*</answer>', response, re.IGNORECASE)
        if match:
            return match.group(1).upper()
        print(response)
        print("MALFORMATTED RESPONSE!")
        return self.default_label
    
    def _format_doc_inputs(self, docs: List[str]) -> str:
        """
        Format a list of document texts into an XML string with enumerated document IDs.
        Each document is wrapped in a <document id=X> ... </document> block.
        """
        formatted_docs = []
        for i, text in enumerate(docs):
            formatted_docs.append(f"<document id={i}>\n{text}\n</document>")
        return "\n".join(formatted_docs)
    
    def score(self, query: str, doc: str) -> float:
        """Score a single document."""
        # Format the single document as an XML input.
        doc_xml = self._format_doc_inputs([doc])
        prompt = self.prompt_template.format(query=query, docu_inputs=doc_xml)
        response = self._get_completion(prompt)
        print(response)
        answer = self._parse_response(response)
        print(answer)
        return 1.0 if answer == "RELEVANT" else 0.0
        
    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        """Rank a list of documents based on relevance to query."""
        docs = prep_docs(docs, doc_ids, metadata)
        doc_texts = [doc.text for doc in docs]
        # Format all document texts into one XML block.
        docs_xml = self._format_doc_inputs(doc_texts)
        prompt = self.prompt_template.format(query=query, docu_inputs=docs_xml)
        response = self._get_completion(prompt)
        print(response)

        pattern = re.compile(r'<document id=(\d+)>(.*?)</document>', re.DOTALL)
        matches = pattern.findall(response)
        doc_scores = {}
        for doc_id, content in matches:
            ans = self._parse_response(content)
            doc_scores[int(doc_id)] = 1.0 if ans == "RELEVANT" else 0.0
        
        # Preserve original order while sorting by score descending.
        scores_with_index = []
        for i, doc in enumerate(docs):
            score = doc_scores.get(i, 0.0)
            scores_with_index.append((score, i, doc))
            
        scores_with_index.sort(key=lambda x: (-x[0], x[1]))
        
        ranked_results = [
            Result(document=doc, score=score, rank=idx + 1)
            for idx, (score, _, doc) in enumerate(scores_with_index)
        ]
        
        return RankedResults(results=ranked_results, query=query, has_scores=True)
