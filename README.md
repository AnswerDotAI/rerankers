
# rerankers

![Python Versions](https://img.shields.io/badge/Python-3.8_3.9_3.10_3.11-blue)
[![Downloads](https://static.pepy.tech/badge/rerankers/month)](https://pepy.tech/project/rerankers)
[![Twitter Follow](https://img.shields.io/twitter/follow/bclavie?style=social)](https://twitter.com/bclavie)


_A lightweight unified API for various reranking models. Developed by [@bclavie](https://twitter.com/bclavie) as a member of [answer.ai](https://www.answer.ai)_

---

Welcome to `rerankers`! Our goal is to provide users with a simple API to use any reranking models.

## Updates

- v0.1.2: 🆕 Voyage reranking API
- v0.1.1: Langchain integration fixed!
- v0.1.0: Initial release

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
ranker = Reranker('mixedbread-ai/mxbai-rerank-xlarge-v1', model_type='cross-encoder')

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

# ColBERTv2 reranker
ranker = Reranker("colbert")

# ... Or a non-default colbert model:
ranker = Reranker(model_name_or_path, model_type = "colbert")

```

_Rerankers will always try to infer the model you're trying to use based on its name, but it's always safer to pass a `model_type` argument to it if you can!_

Then, regardless of which reranker is loaded, use the loaded model to rank a query against documents:

```python
> results = ranker.rank(query="I love you", docs=["I hate you", "I really like you"], doc_ids=[0,1])
> results
RankedResults(results=[Result(doc_id=1, text='I really like you', score=0.26170814, rank=1), Result(doc_id=0, text='I hate you', score=0.079210326, rank=2)], query='I love you', has_scores=True)
```

You don't need to pass `doc_ids`! If not provided, they'll be auto-generated as integers corresponding to the index of a document in `docs`.

All rerankers will return a `RankedResults` object, which is a pydantic object containing a list of `Result` objects and some other useful information, such as the original query. You can retrieve the top `k` results from it by running `top_k()`:

```python
> results.top_k(1)
[Result(doc_id=1, text='I really like you', score=0.26170814, rank=1)]
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
- 🟠 RankGPT (Implemented using original repo, but missing the rankllm's repo improvements)
- ✅ T5-based pointwise rankers (InRanker, MonoT5...)
- ✅ Cohere API rerankers
- ✅ Jina API rerankers
- 🟠 ColBERT-based reranker - not a model initially designed for reranking, but quite strong (Implementation could be optimised and is from a third-party implementation.)
- 📍 MixedBread API (Reranking API not yet released)
- 📍⭐ RankLLM/RankZephyr (Proper RankLLM implementation will replace the RankGPT one, and introduce RankZephyr support)
- 📍 LiT5

Features:
- ✅ Reranking 
- ✅ Consistency notebooks to ensure performance on `scifact` matches the litterature for any given model implementation (Except RankGPT, where results are harder to reproduce).
- 📍 Training on Python >=3.10 (via interfacing with other libraries)
- 📍 ONNX runtime support --> Unlikely to be immediate
- ❌(📍Maybe?) Training via rerankers directly