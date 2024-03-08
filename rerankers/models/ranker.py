from abc import ABC, abstractmethod
from typing import List, Optional, Union
from rerankers.results import RankedResults


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
        docs: List[str],
        doc_ids: Optional[Union[List[str], str]] = None,
    ) -> RankedResults:
        """
        End-to-end reranking of documents.
        """
        pass

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
