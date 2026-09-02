"""Similarity metric normalization utilities for RAG results."""
import math

def to_percentage(raw_score: float, metric: str = "cosine") -> float:
    """Normalize raw distance/similarity scores to an interpretable percentage (0.0% - 100.0%).
    
    Args:
        raw_score: The raw numeric score from the vector database.
        metric: Metric type - 'cosine', 'dot_product', 'euclidean' (or 'l2').
        
    Returns:
        float: Percentage value rounded to 2 decimal places (0.0 to 100.0).
    """
    m = metric.lower().strip()
    
    if m == "cosine":
        # Cosine similarity ranges from -1.0 to 1.0 (or 0.0 to 1.0 for non-negative embeddings)
        # We rescale [-1, 1] linearly to [0, 100]
        clamped = max(-1.0, min(1.0, float(raw_score)))
        pct = ((clamped + 1.0) / 2.0) * 100.0
        return round(max(0.0, min(100.0, pct)), 2)
        
    elif m in ("dot", "dot_product", "inner_product"):
        # Sigmoid transform for unbounded dot products
        try:
            sigmoid = 1.0 / (1.0 + math.exp(-float(raw_score)))
            return round(sigmoid * 100.0, 2)
        except OverflowError:
            return 100.0 if raw_score > 0 else 0.0
            
    elif m in ("euclidean", "l2"):
        # Euclidean distance ranges from 0 to inf. Similarity = 1 / (1 + distance)
        dist = max(0.0, float(raw_score))
        sim = 1.0 / (1.0 + dist)
        return round(sim * 100.0, 2)
        
    else:
        # Default fallback clamp
        return round(max(0.0, min(100.0, float(raw_score) * 100.0 if raw_score <= 1.0 else 100.0)), 2)
