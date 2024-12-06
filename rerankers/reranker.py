from typing import Optional
import warnings
from rerankers.models import AVAILABLE_RANKERS
from rerankers.models.ranker import BaseRanker
from rerankers.utils import vprint

DEFAULTS = {
    "jina": {"en": "jina-reranker-v1-base-en"},
    "pinecone": {"en": "pinecone-rerank-v0"},
    "cohere": {"en": "rerank-english-v3.0", "other": "rerank-multilingual-v3.0"},
    "voyage": {"en": "rerank-lite-1"},
    "mixedbread.ai": {"en": "mixedbread-ai/mxbai-rerank-large-v1"},
    "cross-encoder": {
        "en": "mixedbread-ai/mxbai-rerank-base-v1",
        "fr": "antoinelouis/crossencoder-camembert-base-mmarcoFR",
        "zh": "BAAI/bge-reranker-base",
        "other": "corrius/cross-encoder-mmarco-mMiniLMv2-L12-H384-v1",
    },
    "t5": {"en": "unicamp-dl/InRanker-base", "other": "unicamp-dl/mt5-base-mmarco-v2"},
    "lit5": {
        "en": "castorini/LiT5-Distill-base",
    },
    "rankgpt": {"en": "gpt-4-turbo-preview", "other": "gpt-4-turbo-preview"},
    "rankgpt3": {"en": "gpt-3.5-turbo", "other": "gpt-3.5-turbo"},
    "rankgpt4": {"en": "gpt-4", "other": "gpt-4"},
    "rankllm": {"en": "gpt-4o", "other": "gpt-4o"},
    "colbert": {
        "en": "colbert-ir/colbertv2.0",
        "fr": "bclavie/FraColBERTv2",
        "ja": "bclavie/JaColBERTv2",
        "es": "AdrienB134/ColBERTv2.0-spanish-mmarcoES",
    },
    "flashrank": {"en": "ms-marco-MiniLM-L-12-v2", "other": "ms-marco-MultiBERT-L-12"},
    "text-embeddings-inference": {"other": "BAAI/bge-reranker-base"},
    "llm-layerwise": {
        "en": "BAAI/bge-reranker-v2.5-gemma2-lightweight",
        "other": "BAAI/bge-reranker-v2.5-gemma2-lightweight",
    },
    "monovlm": {
        "en": "lightonai/MonoQwen2-VL-v0.1",
        "other": "lightonai/MonoQwen2-VL-v0.1"
    }
}

DEPS_MAPPING = {
    "TransformerRanker": "transformers",
    "T5Ranker": "transformers",
    "LiT5Ranker": "lit5",
    "RankGPTRanker": "gpt",
    "APIRanker": "api",
    "ColBERTRanker": "transformers",
    "FlashRankRanker": "flashrank",
    "RankLLMRanker": "rankllm",
    "LLMLayerWiseRanker": "transformers",
    "MonoVLMRanker": "transformers"
}

PROVIDERS = ["cohere", "jina", "voyage", "mixedbread.ai", "pinecone", "text-embeddings-inference"]


def _get_api_provider(model_name: str, model_type: Optional[str] = None) -> str:
    if model_type in PROVIDERS or any(provider in model_name for provider in PROVIDERS):
        return model_type or next(
            (provider for provider in PROVIDERS if provider in model_name), None
        )
    # Check if the model_name is a key in DEFAULTS to set the provider correctly
    return next(
        (
            provider
            for provider in PROVIDERS
            if model_name in DEFAULTS
            and any(provider in values for values in DEFAULTS[model_name].values())
        ),
        None,
    )


def _get_model_type(model_name: str, explicit_model_type: Optional[str] = None) -> str:
    if explicit_model_type:
        model_mapping = {
            "cohere": "APIRanker",
            "pinecone": "APIRanker",
            "jina": "APIRanker",
            "voyage": "APIRanker",
            "text-embeddings-inference": "APIRanker",
            "rankgpt": "RankGPTRanker",
            "lit5": "LiT5Ranker",
            "t5": "T5Ranker",
            "colbert": "ColBERTRanker",
            "cross-encoder": "TransformerRanker",
            "flashrank": "FlashRankRanker",
            "rankllm": "RankLLMRanker",
            "llm-layerwise": "LLMLayerWiseRanker",
            "monovlm": "MonoVLMRanker"
        }
        return model_mapping.get(explicit_model_type, explicit_model_type)
    else:
        model_name = model_name.lower()
        model_mapping = {
            "lit5": "LiT5Ranker",
            "t5": "T5Ranker",
            "inranker": "T5Ranker",
            "rankllm": "RankLLMRanker",
            "rankgpt": "RankGPTRanker",
            "gpt": "RankGPTRanker",
            "colbert": "ColBERTRanker",
            "cohere": "APIRanker",
            "pinecone": "APIRanker",
            "jina": "APIRanker",
            "voyage": "APIRanker",
            "text-embeddings-inference": "APIRanker",
            "ms-marco-minilm-l-12-v2": "FlashRankRanker",
            "ms-marco-multibert-l-12": "FlashRankRanker",
            "vicuna": "RankLLMRanker",
            "zephyr": "RankLLMRanker",
            "bge-reranker-v2.5-gemma2-lightweight": "LLMLayerWiseRanker",
            "monovlm": "MonoVLMRanker",
            "monoqwen2-vl": "MonoVLMRanker"
        }
        for key, value in model_mapping.items():
            if key in model_name:
                if key == "gpt":
                    warnings.warn(
                        "The key 'gpt' currently defaults to the rough rankGPT implementation. From version 0.0.5 onwards, 'gpt' will default to RankLLM instead. Please specify the 'rankgpt' `model_type` if you want to keep the current behaviour",
                        DeprecationWarning,
                    )
                return value
        if (
            any(
                keyword in model_name
                for keyword in ["minilm", "bert", "cross-encoders/"]
            )
            and "/" in model_name
        ):
            return "TransformerRanker"
        print(
            "Warning: Model type could not be auto-mapped with the defaults list. Defaulting to TransformerRanker."
        )
        print(
            "If your model is NOT intended to be ran as a one-label cross-encoder, please reload it and specify the model_type!",
            "Otherwise, you may ignore this warning. You may specify `model_type='cross-encoder'` to suppress this warning in the future.",
        )
        return "TransformerRanker"


def _get_defaults(
    model_name: str,
    model_type: Optional[str] = None,
    lang: str = "en",
    verbose: int = 1,
) -> str:
    if model_name in DEFAULTS.keys():
        print(f"Loading default {model_name} model for language {lang}")
        try:
            model_name = DEFAULTS[model_name][lang]
        except KeyError:
            if "other" not in DEFAULTS[model_name]:
                print(
                    f"Model family {model_name} does not have a default for language {lang}"
                )
                print(
                    "Aborting now... Please retry with another model family or by specifying a model"
                )
                return None, None
            model_name = DEFAULTS[model_name]["other"]
    model_type = _get_model_type(model_name, model_type)
    vprint(f"Default Model: {model_name}", verbose)

    return model_name, model_type


def Reranker(
    model_name: str,
    lang: str = "en",
    model_type: Optional[str] = None,
    verbose: int = 1,
    **kwargs,
) -> Optional[BaseRanker]:
    original_model_name = model_name
    api_provider = _get_api_provider(model_name, model_type)
    if api_provider or model_name.lower() in PROVIDERS:
        if model_name.lower() in PROVIDERS:
            api_provider = model_name.lower()
            model_type = "APIRanker"
            model_name = (
                DEFAULTS[api_provider][lang]
                if lang in DEFAULTS[api_provider]
                else DEFAULTS[api_provider]["other"]
            )
            print(
                f"Auto-updated model_name to {model_name} for API provider {api_provider}"
            )
        else:
            model_type = "APIRanker"
    else:
        if original_model_name in DEFAULTS.keys():
            model_name, model_type = _get_defaults(
                original_model_name, model_type, lang, verbose
            )
            if model_name is None:
                return None
            api_provider = _get_api_provider(model_name, model_type)
            if api_provider:
                model_type = "APIRanker"

    if api_provider:
        kwargs["api_provider"] = api_provider

    model_type = _get_model_type(model_name, model_type)

    try:
        vprint(f"Loading {model_type} model {model_name} (this message can be suppressed by setting verbose=0)", verbose)
        return AVAILABLE_RANKERS[model_type](model_name, verbose=verbose, **kwargs)
    except KeyError:
        print(
            f"You don't have the necessary dependencies installed to use {model_type}."
        )
        print(
            f'Please install the necessary dependencies for {model_type} by running `pip install "rerankers[{DEPS_MAPPING[model_type]}]"`',
            'or `pip install "rerankers[all]" to install the dependencies for all reranker types.',
        )
        return None
