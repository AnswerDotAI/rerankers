"""`PyTerrier <https://github.com/terrier-org/pyterrier>`__ integration for rerankers.

This module exposes :class:`~rerankers.integrations.pyterrier.RerankersTransformer`, a PyTerrier transformer that
wraps any :class:`~rerankers.models.ranker.BaseRanker` so it can be dropped into a PyTerrier retrieval
pipeline. You normally obtain one by calling
:meth:`~rerankers.models.ranker.BaseRanker.as_pyterrier_transformer` on a loaded reranker.
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import pyterrier as pt

from rerankers.models.ranker import BaseRanker


class RerankersTransformer(pt.Transformer):
    """A PyTerrier transformer that reranks retrieved results using a rerankers model.

    The transformer consumes a PyTerrier result frame (with ``qid``, ``query``, ``docno`` and ``text``
    columns) and reranks the documents independently for each query using the wrapped
    :class:`~rerankers.models.ranker.BaseRanker`. The ``score`` and ``rank`` columns are replaced with the
    values produced by the reranker.

    It is usually placed after a first-stage retriever and a step that attaches the document text (e.g.,
    :func:`pyterrier.text.get_text`). Rather than constructing it directly, call
    :meth:`~rerankers.models.ranker.BaseRanker.as_pyterrier_transformer` on a loaded reranker:

    .. code-block:: python
        :caption: Rerank BM25 results with a cross-encoder

        >>> import pyterrier as pt
        >>> from rerankers import Reranker
        >>> dataset = pt.get_dataset('irds:vaswani')
        >>> index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
        >>> reranker = Reranker('cross-encoder') # loads mixedbread-ai/mxbai-rerank-base-v1
        >>> # Retrieve with BM25, load the document text, then re-rank
        >>> pipeline = index.bm25() >> dataset.text_loader() >> reranker.as_pyterrier_transformer()
        >>> pipeline.search('chemical reactions')
    """

    def __init__(self, ranker: BaseRanker, *, text_field: str = 'text'):
        """Wrap a rerankers model as a PyTerrier transformer.

        Args:
            ranker: The loaded reranker to apply. This is any
                :class:`~rerankers.models.ranker.BaseRanker`, e.g., the object returned by
                :func:`rerankers.Reranker`.
            text_field: The input column holding the document text to rerank. Defaults to ``'text'``.
        """
        self.ranker = ranker
        self.text_field = text_field

    def transform(self, inp: pd.DataFrame) -> pd.DataFrame:
        """Rerank an input result frame.

        Args:
            inp: A result frame with (at least) ``qid``, ``query``, ``docno`` and the text column
                (``text_field``, ``'text'`` by default). The frame is reranked independently for each
                query (grouped by ``qid``).

        Returns:
            pd.DataFrame: A copy of ``inp`` with updated ``score`` and ``rank`` columns, sorted by
            descending score within each query.
        """
        pt.validate.result_frame(inp, ['query', self.text_field])
        inp = inp.reset_index(drop=True)
        scores = {}
        for _, query_frame in inp.groupby('qid', sort=False):
            # Map the scores back to row by index
            results = self.ranker.rank(
                query=query_frame['query'].iloc[0],
                docs=query_frame[self.text_field].tolist(),
                doc_ids=list(query_frame.index),
            )
            if results.has_scores:
                scores.update({r.doc_id: r.score for r in results})
            else:
                # Listwise rerankers (e.g., RankGPT, RankLLM) return a ranking but no scores;
                # fall back to the rank so the best (rank 1) gets the highest score.
                scores.update({r.doc_id: -r.rank for r in results})
        result = inp.assign(score=scores)
        result = pt.model.add_ranks(result)
        return result.sort_values(['qid', 'rank'])

    def schematic(self, *, input_columns: Optional[List[str]] = None) -> Dict[str, Any]:
        # Label the schematic with the name of the reranker class
        return {'label': type(self.ranker).__name__}
