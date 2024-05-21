from abc import ABC, abstractmethod
from asyncio import get_event_loop
from functools import partial
from typing import List, Optional, Union
from rerankers.results import RankedResults
from rerankers.documents import Document


class BaseRanker(ABC):
    @abstractmethod
    def __init__(self, model_name_or_path: str, verbose: int):
        pass

    @abstractmethod
    def score(self, query: str, doc: str) -> float:
        pass

    @abstractmethod
    def rank(
        self,
        query: str,
        docs: Union[str, List[str], Document, List[Document]],
        doc_ids: Optional[Union[List[str], List[int]]] = None,
    ) -> RankedResults:
        """
        End-to-end reranking of documents.
        """
        pass

    async def rank_async(
        self,
        query: str,
        docs: List[str],
        doc_ids: Optional[Union[List[str], str]] = None,
    ) -> RankedResults:


        loop = get_event_loop()
        return await loop.run_in_executor(None, partial(self.rank, query, docs, doc_ids))

    def as_langchain_compressor(self, k: int = 10):
        try:
            from rerankers.integrations.langchain import RerankerLangChainCompressor

            return RerankerLangChainCompressor(model=self, k=k)
        except ImportError:
            print(
                "You need to install langchain and langchain_core to export a reranker as a LangChainCompressor!"
            )
            print(
                'Please run `pip install "rerankers[langchain]"` to get all the required dependencies.'
            )
