AVAILABLE_RANKERS = {}

try:
    from rerankers.models.transformer_ranker import TransformerRanker

    AVAILABLE_RANKERS["TransformerRanker"] = TransformerRanker
except ImportError:
    pass
try:
    from rerankers.models.api_rankers import APIRanker

    AVAILABLE_RANKERS["APIRanker"] = APIRanker
except ImportError:
    pass
try:
    from rerankers.models.rankgpt_rankers import RankGPTRanker

    AVAILABLE_RANKERS["RankGPTRanker"] = RankGPTRanker
except ImportError:
    pass
try:
    from rerankers.models.t5ranker import T5Ranker

    AVAILABLE_RANKERS["T5Ranker"] = T5Ranker
except ImportError:
    pass

try:
    from rerankers.models.colbert_ranker import ColBERTRanker

    AVAILABLE_RANKERS["ColBERTRanker"] = ColBERTRanker
except ImportError:
    pass
