from typing import Union, List, Optional
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result
from rerankers.utils import prep_docs
from rerankers.documents import Document
from string import Template


import requests
import json


URLS = {
    "cohere": "https://api.cohere.ai/v1/rerank",
    "jina": "https://api.jina.ai/v1/rerank",
    "voyage": "https://api.voyageai.com/v1/rerank",
    "mixedbread.ai": "https://api.mixedbread.ai/v1/reranking",
    "pinecone": "https://api.pinecone.io/rerank",
}
AUTHORIZATION_KEY_MAPPING = {
    "pinecone": "Api-Key"
}
API_VERSION_MAPPING = {
    "pinecone": {"X-Pinecone-API-Version": "2024-10"}
}

API_KEY_MAPPING = {
    "pinecone": Template("$api_key")
}

DOCUMENT_KEY_MAPPING = {
    "mixedbread.ai": "input",
    "text-embeddings-inference":"texts"
}
RETURN_DOCUMENTS_KEY_MAPPING = {
    "mixedbread.ai":"return_input",
    "text-embeddings-inference":"return_text"
}
RESULTS_KEY_MAPPING = {
    "voyage": "data",
    "mixedbread.ai": "data",
    "pinecone": "data",
    "text-embeddings-inference": None
}
SCORE_KEY_MAPPING = {
    "mixedbread.ai": "score",
    "pinecone": "score",
    "text-embeddings-inference":"score"
}

class APIRanker(BaseRanker):
    def __init__(self, model: str, api_key: str, api_provider: str, verbose: int = 1, url: str = None):
        self.api_provider = api_provider.lower()
        self.api_key = API_KEY_MAPPING.get(self.api_provider,Template("Bearer $api_key")).substitute(api_key=api_key)
        authorization_key = AUTHORIZATION_KEY_MAPPING.get(self.api_provider,"Authorization")
        api_version = API_VERSION_MAPPING.get(self.api_provider,None)
        self.model = model
        self.verbose = verbose
        self.ranking_type = "pointwise"
        self.headers = {
            "accept": "application/json",
            "content-type": "application/json",
            authorization_key: self.api_key,
        }
        if api_version:
            self.headers.update(api_version)
        self.url = url if url else URLS[self.api_provider]


    def _get_document_text(self, r: dict) -> str:
        if self.api_provider == "voyage":
            return r["document"]
        elif self.api_provider == "mixedbread.ai":
            return r["input"]
        elif self.api_provider == "text-embeddings-inference":
            return r["text"]
        else:
            return r["document"]["text"]

    def _get_score(self, r: dict) -> float:
        score_key = SCORE_KEY_MAPPING.get(self.api_provider,"relevance_score")
        return r[score_key]

    def _parse_response(
        self, response: dict,  docs: List[Document],
    ) -> RankedResults:
        ranked_docs = []
        results_key = RESULTS_KEY_MAPPING.get(self.api_provider,"results")

        for i, r in enumerate(response[results_key] if results_key else response):
            ranked_docs.append(
                Result(
                    document=docs[r["index"]],
                    text=self._get_document_text(r),
                    score=self._get_score(r),
                    rank=i + 1,
                )
            )

        return ranked_docs

    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
        metadata: Optional[List[dict]] = None,
    ) -> RankedResults:
        docs = prep_docs(docs, doc_ids, metadata)
        payload = self._format_payload(query, docs)
        response = requests.post(self.url, headers=self.headers, data=payload)
        results = self._parse_response(response.json(), docs)
        return RankedResults(results=results, query=query, has_scores=True)


    def _format_payload(self, query: str, docs: List[str]) -> str:
        top_key = (
            "top_n" if self.api_provider not in ["voyage", "mixedbread.ai"] else "top_k"
        )
        documents_key = DOCUMENT_KEY_MAPPING.get(self.api_provider,"documents")
        return_documents_key = RETURN_DOCUMENTS_KEY_MAPPING.get(self.api_provider,"return_documents")

        documents = (
            [d.text for d in docs] if self.api_provider not in ["pinecone"] else [{"text": d.text} for d in docs]
        ) 
        
        payload = {
            "model": self.model,
            "query": query,
            documents_key: documents,
            top_key: len(docs),
            return_documents_key: True,
        }
        return json.dumps(payload)

    def score(self, query: str, doc: str) -> float:
        payload = self._format_payload(query, [doc])
        response = requests.post(self.url, headers=self.headers, data=payload)
        results = self._parse_response(response.json(), [doc])
        return results[0].score
