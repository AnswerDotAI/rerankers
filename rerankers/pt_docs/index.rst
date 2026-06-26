Rerankers + PyTerrier
=====================================

`rerankers <https://github.com/answerdotai/rerankers>`__ provides a single, unified API over many different
reranking models -- cross-encoders, T5 rerankers, ColBERT, LLM-based rerankers (RankGPT, RankLLM), and
API-based rerankers (Cohere, Jina, Voyage, Pinecone, ...), among others. Its PyTerrier integration lets you
use any of these rerankers in a `PyTerrier <https://github.com/terrier-org/pyterrier>`__ pipeline.


Quick Start
-------------------------------------

You can install ``rerankers`` with its PyTerrier dependencies using pip:

.. code-block:: console
   :caption: Install ``rerankers`` with PyTerrier support

   $ pip install "rerankers[pyterrier]"

Most reranker families require extra dependencies (e.g., ``transformers`` for cross-encoders). See the
`rerankers README <https://github.com/answerdotai/rerankers>`__ for the full list of extras, or install
everything at once with ``pip install "rerankers[all,pyterrier]"``.


Reranking in a PyTerrier Pipeline
-------------------------------------

Load a reranker with :func:`rerankers.Reranker` and convert it into a PyTerrier transformer by calling
:meth:`~rerankers.models.ranker.BaseRanker.as_pyterrier_transformer`. The transformer reranks the input
results for each query, so it is typically placed after first-stage retrieval and text loading:

.. code-block:: python
   :caption: Rerank BM25 results with a cross-encoder

   >>> import pyterrier as pt
   >>> from rerankers import Reranker
   >>> dataset = pt.get_dataset('irds:vaswani')
   >>> index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
   >>> reranker = Reranker('cross-encoder')  # loads mixedbread-ai/mxbai-rerank-base-v1
   >>> # Retrieve with BM25, load the document text, then re-rank
   >>> pipeline = index.bm25() >> dataset.text_loader() >> reranker.as_pyterrier_transformer()
   >>> pipeline.search('chemical reactions')

.. schematic::
   from rerankers import Reranker
   dataset = pt.get_dataset('irds:vaswani')
   index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
   reranker = Reranker('cross-encoder')
   index.bm25() >> dataset.text_loader() >> reranker.as_pyterrier_transformer()

.. note::

   The reranker transformer expects ``qid``, ``query``, ``docno`` and ``text`` columns in its input. Be sure
   to load the document's text (e.g., using the dataset's ``text_loader()`` or
   :func:`pyterrier.text.get_text`) before the reranker.

How-To Guides
-------------------------------------

The following guides answer common questions about using rerankers in PyTerrier.

.. how-to:: How do I re-rank only the top-ranked documents?

    .. _rerankers:how-to:cutoff:

    Neural re-rankers are far slower than first-stage retrieval, so you usually only re-rank the most
    promising documents. Apply a rank cutoff (``%``) before the re-ranker so it scores just the top
    results for each query:

    .. code-block:: python
        :caption: Re-rank only the top 100 BM25 results

        import pyterrier as pt
        from rerankers import Reranker

        dataset = pt.get_dataset('irds:vaswani')
        index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
        reranker = Reranker('cross-encoder').as_pyterrier_transformer()

        pipeline = index.bm25() % 100 >> dataset.text_loader() >> reranker # :footnote: ``% 100`` keeps only the top 100 documents per query before re-ranking. Lower it for faster (but shallower) re-ranking.


.. how-to:: How do I use a different kind of re-ranker?

    .. _rerankers:how-to:model-types:

    ``rerankers`` exposes many model families behind the same :func:`rerankers.Reranker` API. Change the
    model you load and the rest of the pipeline is unchanged, because every reranker becomes the same kind
    of transformer.

    .. code-block:: python
        :caption: Load different re-ranker families

        from rerankers import Reranker

        cross_encoder = Reranker('cross-encoder').as_pyterrier_transformer() # :footnote: A sequence-classification cross-encoder (loads ``mixedbread-ai/mxbai-rerank-base-v1`` by default).
        monot5 = Reranker('t5').as_pyterrier_transformer() # :footnote: A monoT5-style sequence-to-sequence re-ranker.
        colbert = Reranker('colbert').as_pyterrier_transformer() # :footnote: A late-interaction (ColBERT) re-ranker.

    You can also load an explicit model, e.g. ``Reranker('mixedbread-ai/mxbai-rerank-large-v1',
    model_type='cross-encoder')``. See the `rerankers README
    <https://github.com/answerdotai/rerankers>`__ for the full list of supported models.


.. how-to:: How do I use an API-based re-ranker?

    .. _rerankers:how-to:api:

    Hosted re-rankers (Cohere, Jina, Voyage, Pinecone, ...) work the same way -- pass the provider name and
    your API key:

    .. code-block:: python
        :caption: Re-rank with a hosted (API) model

        from rerankers import Reranker

        reranker = Reranker('cohere', api_key='YOUR_API_KEY').as_pyterrier_transformer() # :footnote: Swap ``'cohere'`` for another provider such as ``'jina'`` or ``'voyage'``.


.. how-to:: How do I evaluate a re-ranking pipeline?

    .. _rerankers:how-to:evaluate:

    Because the re-ranker is an ordinary PyTerrier transformer, you can score it with
    :func:`pyterrier.Experiment`, comparing first-stage retrieval against the re-ranked pipeline:

    .. code-block:: python
        :caption: Compare BM25 with and without re-ranking

        import pyterrier as pt
        from rerankers import Reranker
        from pyterrier.measures import nDCG, RR

        dataset = pt.get_dataset('irds:vaswani')
        index = pt.Artifact.from_hf('pyterrier/vaswani.terrier')
        reranked = index.bm25() % 100 >> dataset.text_loader() >> Reranker('cross-encoder').as_pyterrier_transformer()

        pt.Experiment(
            [index.bm25(), reranked], # :footnote: The first run is plain BM25; the second is BM25 followed by the cross-encoder.
            dataset.get_topics(),
            dataset.get_qrels(),
            eval_metrics=[nDCG @ 10, RR @ 10],
            names=['BM25', 'BM25 >> cross-encoder'],
        )


API Documentation
----------------------------------------

Use :meth:`~rerankers.models.ranker.BaseRanker.as_pyterrier_transformer` to build a transformer from any
loaded reranker.

.. automethod:: rerankers.models.ranker.BaseRanker.as_pyterrier_transformer

The method returns a :class:`~rerankers.integrations.pyterrier.RerankersTransformer` transformer:

.. autoclass:: rerankers.integrations.pyterrier.RerankersTransformer
   :members:
   :show-inheritance:


Acknowledgements
-------------------------------------

The PyTerrier integration is part of the `rerankers <https://github.com/answerdotai/rerankers>`__ library
by Benjamin Clavié and contributors. Check out the GitHub repository for
`a full list of contributors <https://github.com/answerdotai/rerankers/graphs/contributors>`__.
