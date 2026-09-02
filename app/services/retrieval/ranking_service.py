"""Local re-ranking service using FlashRank (100% offline & local)."""
from typing import List
# pyrefly: ignore [missing-import]
from flashrank import Ranker, RerankRequest
from app.services.retrieval.vectorstore.base import SearchResult
from app.services.retrieval.similarity import to_percentage
from app.config import settings

class RankingService:
    """Local reranking engine using FlashRank cross-encoder."""
    
    # ponytail: ms-marco-TinyBERT-L-2-v2 for sub-5ms CPU reranking, upgrade to ms-marco-MiniLM-L-6-v2 if deeper rank precision needed
    def __init__(self, model_name: str = "ms-marco-TinyBERT-L-2-v2"):
        self.model_name = model_name
        self._ranker = Ranker(model_name=self.model_name, cache_dir="./data/models")

    def rerank(self, query: str, results: List[SearchResult], top_k: int = 5) -> List[SearchResult]:
        if not results or not settings.ENABLE_RERANKING:
            return results[:top_k]
            
        passages = [
            {"id": r.chunk_id, "text": r.text, "meta": r.metadata, "original_result": r}
            for r in results
        ]
        req = RerankRequest(query=query, passages=passages)
        reranked = self._ranker.rerank(req)
        
        output = []
        for rank, item in enumerate(reranked[:top_k], start=1):
            raw_score = float(item.get("score", 0.0))
            pct = to_percentage(raw_score, metric="cosine")
            output.append(SearchResult(
                chunk_id=item.get("id"), text=item.get("text"),
                metadata=item.get("meta", {}), raw_score=round(raw_score, 4),
                similarity_pct=pct, rank=rank
            ))
        return output

ranking_service = RankingService()
