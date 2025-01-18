from rerankers.results import RankedResults
from rerankers.models.ranker import BaseRanker
from rerankers.results import RankedResults, Result
from fastembed.rerank.cross_encoder import TextCrossEncoder
from rerankers.utils import prep_docs


class FastEmbedRanker(BaseRanker):

    def __init__(self, model_name_or_path, verbose=None):

        self.model = TextCrossEncoder(model_name=model_name_or_path)

    def rank(self, query, docs):
        docs = prep_docs(docs)
        scores = list(self.model.rerank(query, [d.text for d in docs]))
        indices = sorted(range(len(scores)), key=lambda k: scores[k], reverse=True)

        ranked_results = [
            Result(document=docs[idx], score=scores[idx], rank=i + 1)
            for i, idx in enumerate(indices)
        ]

        return RankedResults(results=ranked_results, query=query, has_scores=True)

    def score(self, query, doc):
        score = list(self.model.rerank(query, [doc]))[0]
        return score
