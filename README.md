
# rerankers

![Python Versions](https://img.shields.io/badge/Python-3.8_3.9_3.10_3.11-blue)
[![Downloads](https://static.pepy.tech/badge/rerankers/month)](https://pepy.tech/project/rerankers)
[![Twitter Follow](https://img.shields.io/twitter/follow/bclavie?style=social)](https://twitter.com/bclavie)


_A lightweight unified API for various reranking models. Developed by [@bclavie](https://twitter.com/bclavie) as a member of [answer.ai](https://www.answer.ai)_

---

Welcome to `rerankers`! Our goal is to provide users with a simple API to use any reranking models.

## Recent Updates
_A longer release history can be found in the [Release History](#release-history) section of this README._

- v0.5.2: Minor ColBERT fixes
- v0.5.1: Minor change making RankedResults subscribable, meaning results[0] will return the result for the first document, etc... ⚠️ This is sorted by **passed document order**, not by results, you should use `.top_k()` to get sorted results!
- v0.5.0: Added support for the current state-of-the-art rerankers, BAAI's series of `BGE` layerwise LLM rerankers, based on [Gemma](https://huggingface.co/BAAI/bge-reranker-v2.5-gemma2-lightweight) and MiniCPM. These are different from RankGPT, as they're not listwise: the models are repurposed as "cross-encoders", and do output logit scores.

## Why `rerankers`?

Rerankers are an important part of any retrieval architecture, but they're also often more obscure than other parts of the pipeline.

Sometimes, it can be hard to even know which one to use. Every problem is different, and the best model for use X is not necessarily the same one as for use Y.

Moreover, new reranking methods keep popping up: for example, RankGPT, using LLMs to rerank documents, appeared just last year, with very promising zero-shot benchmark results.

All the different reranking approaches tend to be done in their own library, with varying levels of documentation. This results in an even higher barrier to entry. New users are required to swap between multiple unfamiliar input/output formats, all with their own quirks!

`rerankers` seeks to address this problem by providing a simple API for all popular rerankers, no matter the architecture.

`rerankers` aims to be:
- 🪶 Lightweight. It ships with only the bare necessities as dependencies.
- 📖 Easy-to-understand. There's just a handful of calls to learn, and you can then use the full range of provided reranking models.
- 🔗 Easy-to-integrate. It should fit in just about any existing pipelines, with only a few lines of code!
- 💪 Easy-to-expand. Any new reranking models can be added with very little knowledge of the codebase. All you need is a new class with a `rank()` function call mapping a (query, [documents]) input to a `RankedResults` output.
- 🐛 Easy-to-debug. This is a beta release and there might be issues, but the codebase is conceived in such a way that most issues should be easy to track and fix ASAP.

## Get Started

Installation is very simple. The core package ships with just two dependencies, `tqdm` and `pydantic`, so as to avoid any conflict with your current environment.
You may then install only the dependencies required by the models you want to try out:

```sh
# Core package only, will require other dependencies already installed
pip install rerankers

# All transformers-based approaches (cross-encoders, t5, colbert)
pip install "rerankers[transformers]"

# RankGPT
pip install "rerankers[gpt]"

# API-based rerankers (Cohere, Jina, soon MixedBread)
pip install "rerankers[api]"

# FlashRank rerankers (ONNX-optimised, very fast on CPU)
pip install "rerankers[flashrank]"

# RankLLM rerankers (better RankGPT + support for local models such as RankZephyr and RankVicuna)
# Note: RankLLM is only supported on Python 3.10+! This will not work with Python 3.9
pip install "rerankers[rankllm]"

# To support LLM-Layerwise rerankers (which need flash-attention installed)
pip install "rerankers[llmlayerwise]"

# All of the above
pip install "rerankers[all]"
```

## Usage

Load any supported reranker in a single line, regardless of the architecture:
```python
from rerankers import Reranker

# Cross-encoder default. You can specify a 'lang' parameter to load a multilingual version!
ranker = Reranker('cross-encoder')

# Specific cross-encoder
ranker = Reranker('mixedbread-ai/mxbai-rerank-large-v1', model_type='cross-encoder')

# FlashRank default. You can specify a 'lang' parameter to load a multilingual version!
ranker = Reranker('flashrank')

# Specific flashrank model.
ranker = Reranker('ce-esci-MiniLM-L12-v2', model_type='flashrank')

# Default T5 Seq2Seq reranker
ranker = Reranker("t5")

# Specific T5 Seq2Seq reranker
ranker = Reranker("unicamp-dl/InRanker-base", model_type = "t5")

# API (Cohere)
ranker = Reranker("cohere", lang='en' (or 'other'), api_key = API_KEY)

# Custom Cohere model? No problem!
ranker = Reranker("my_model_name", api_provider = "cohere", api_key = API_KEY)

# API (Jina)
ranker = Reranker("jina", api_key = API_KEY)

# RankGPT4-turbo
ranker = Reranker("rankgpt", api_key = API_KEY)

# RankGPT3-turbo
ranker = Reranker("rankgpt3", api_key = API_KEY)

# RankGPT with another LLM provider
ranker = Reranker("MY_LLM_NAME" (check litellm docs), model_type = "rankgpt", api_key = API_KEY)

# RankLLM with default GPT (GPT-4o)
ranker = Reranker("rankllm", api_key = API_KEY)

# RankLLM with specified GPT models
ranker = Reranker('gpt-4-turbo', model_type="rankllm", api_key = API_KEY)

# ColBERTv2 reranker
ranker = Reranker("colbert")

# LLM Layerwise Reranker
ranker = Reranker('llm-layerwise')

# ... Or a non-default colbert model:
ranker = Reranker(model_name_or_path, model_type = "colbert")

```

_Rerankers will always try to infer the model you're trying to use based on its name, but it's always safer to pass a `model_type` argument to it if you can!_

Then, regardless of which reranker is loaded, use the loaded model to rank a query against documents:

```python
> results = ranker.rank(query="I love you", docs=["I hate you", "I really like you"], doc_ids=[0,1])
> results
RankedResults(results=[Result(document=Document(text='I really like you', doc_id=1), score=-2.453125, rank=1), Result(document=Document(text='I hate you', doc_id=0), score=-4.14453125, rank=2)], query='I love you', has_scores=True)
```

You don't need to pass `doc_ids`! If not provided, they'll be auto-generated as integers corresponding to the index of a document in `docs`.


You're free to pass metadata too, and it'll be stored with the documents. It'll also be accessible in the results object:

```python
> results = ranker.rank(query="I love you", docs=["I hate you", "I really like you"], doc_ids=[0,1], metadata=[{'source': 'twitter'}, {'source': 'reddit'}])
> results
RankedResults(results=[Result(document=Document(text='I really like you', doc_id=1, metadata={'source': 'twitter'}), score=-2.453125, rank=1), Result(document=Document(text='I hate you', doc_id=0, metadata={'source': 'reddit'}), score=-4.14453125, rank=2)], query='I love you', has_scores=True)
```

If you'd like your code to be a bit cleaner, you can also directly construct `Document` objects yourself, and pass those instead. In that case, you don't need to pass separate `doc_ids` and `metadata`:

```python
> from rerankers import Document
> docs = [Document(text="I really like you", doc_id=0, metadata={'source': 'twitter'}), Document(text="I hate you", doc_id=1, metadata={'source': 'reddit'})]
> results = ranker.rank(query="I love you", docs=docs)
> results
RankedResults(results=[Result(document=Document(text='I really like you', doc_id=0, metadata={'source': 'twitter'}), score=-2.453125, rank=1), Result(document=Document(text='I hate you', doc_id=1, metadata={'source': 'reddit'}), score=-4.14453125, rank=2)], query='I love you', has_scores=True)
```

You can also use `rank_async`, which is essentially just a wrapper to turn `rank()` into a coroutine. The result will be the same:

```python
> results = await ranker.rank_async(query="I love you", docs=["I hate you", "I really like you"], doc_ids=[0,1])
> results
RankedResults(results=[Result(document=Document(text='I really like you', doc_id=1, metadata={'source': 'twitter'}), score=-2.453125, rank=1), Result(document=Document(text='I hate you', doc_id=0, metadata={'source': 'reddit'}), score=-4.14453125, rank=2)], query='I love you', has_scores=True)
```

All rerankers will return a `RankedResults` object, which is a pydantic object containing a list of `Result` objects and some other useful information, such as the original query. You can retrieve the top `k` results from it by running `top_k()`:

```python
> results.top_k(1)
[Result(Document(doc_id=1, text='I really like you', metadata={}), score=0.26170814, rank=1)]
```

The Result objects are transparent when trying to access the documents they store, as `Document` objects simply exist as an easy way to store IDs and metadata. If you want to access a given result's text or metadata, you can directly access it as a property:

```python
> results.top_k(1)[0].text
'I really like you'
```

And that's all you need to know to get started quickly! Check out the overview notebook for more information on the API and the different models, or the langchain example to see how to integrate this in your langchain pipeline.


## Features

Legend:
- ✅ Supported
- 🟠 Implemented, but not fully fledged
- 📍 Not supported but intended to be in the future
- ⭐ Same as above, but **important**.
- ❌ Not supported & not currently planned

Models:
- ✅ Any standard SentenceTransformer or Transformers cross-encoder
- ✅ RankGPT (Available both via the original RankGPT implementation and the improved RankLLM one)
- ✅ T5-based pointwise rankers (InRanker, MonoT5...)
- ✅ LLM-based pointwise rankers (BAAI/bge-reranker-v2.5-gemma2-lightweight, etc...)
- ✅ Cohere, Jina, Voyage and MixedBread API rerankers
- ✅ [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) rerankers (ONNX-optimised models, very fast on CPU)
- ✅ ColBERT-based reranker - not a model initially designed for reranking, but does perform quite strongly in some cases. Implementation is lightweight, based only on transformers.
- 🟠⭐ RankLLM/RankZephyr: supported by wrapping the [rank-llm library](https://github.com/castorini/rank_llm) library! Support for RankZephyr/RankVicuna is untested, but RankLLM + GPT models fully works!
- 📍 LiT5

Features:
- ✅ Metadata!
- ✅ Reranking 
- ✅ Consistency notebooks to ensure performance on `scifact` matches the litterature for any given model implementation (Except RankGPT, where results are harder to reproduce).
- ✅ ONNX runtime support --> Offered through [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) -- in line with the philosophy of the lib, we won't reinvent the wheel when @PrithivirajDamodaran is doing amazing work!
- 📍 Training on Python >=3.10 (via interfacing with other libraries)
- ❌(📍Maybe?) Training via rerankers directly

## Reference

If rerankers has been useful to you in academic work, please do feel free to cite the work below!

```
@misc{clavié2024rerankers,
      title={rerankers: A Lightweight Python Library to Unify Ranking Methods}, 
      author={Benjamin Clavié},
      year={2024},
      eprint={2408.17344},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2408.17344}, 
}
```

## Release History

- v0.4.0: ColBERT performance improvement! It should now be faster and result in stronger results following implementation of the JaColBERTv2.5 dynamic query length method. This version also now supports HuggingFace's Text-Embedding-Server (TEI) inference as an API reranker option, thanks to [@srisudarsan](https://github.com/srisudarsan).
- v0.3.1: T5 bugfix and native default support for new Portuguese T5 rerankers.
- v0.3.0: Many changes! Experimental support for RankLLM, directly backed by the [rank-llm library](https://github.com/castorini/rank_llm). A new `Document` object, courtesy of joint-work by [@bclavie](https://github.com/bclavie) and [Anmol6](https://github.com/Anmol6). This object is transparent, but now offers support for `metadata` stored alongside each document. Many small QoL changes (RankedResults can be itered on directly...)
- v0.2.0: [FlashRank](https://github.com/PrithivirajDamodaran/FlashRank) rerankers, Basic async support thanks to [@tarunamasa](https://github.com/tarunamasa), MixedBread.ai reranking API
- v0.1.2: Voyage reranking API
- v0.1.1: Langchain integration fixed!
- v0.1.0: Initial release
