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

    def as_pyterrier_transformer(self, *, text_field: str = 'text'):
        """Create a `PyTerrier <https://github.com/terrier-org/pyterrier>`__ transformer that reranks with this model.

        The returned transformer can be inserted into a PyTerrier pipeline to rerank a set of retrieved
        results. It expects an input frame with ``query``, ``docno`` and ``text`` columns and replaces the
        ``score`` and ``rank`` columns with the values produced by this reranker.

        Args:
            text_field: The input column holding the document text to rerank. Defaults to ``'text'``.

        Returns:
            :class:`~rerankers.integrations.pyterrier.RerankersTransformer`: A PyTerrier transformer wrapping this model.

        .. code-block:: python
            :caption: Rerank BM25 results in a PyTerrier pipeline

            >>> import pyterrier as pt
            >>> from rerankers import Reranker
            >>> dataset = pt.get_dataset('irds:vaswani')
            >>> index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
            >>> reranker = Reranker('cross-encoder')  # loads mixedbread-ai/mxbai-rerank-base-v1
            >>> # Retrieve with BM25, load the document text, then re-rank
            >>> pipeline = index.bm25() >> dataset.text_loader() >> reranker.as_pyterrier_transformer()
            >>> pipeline.search('chemical reactions')

        .. note::
            This requires PyTerrier to be installed, e.g., via ``pip install "rerankers[pyterrier]"``.
        """
        try:
            from rerankers.integrations.pyterrier import RerankersTransformer

            return RerankersTransformer(self, text_field=text_field)
        except ImportError:
            print(
                "You need to install pyterrier to export a reranker as a PyTerrier Transformer!"
            )
            print(
                'Please run `pip install "rerankers[pyterrier]"` to get all the required dependencies.'
            )
