"""Fixed size character/token sliding window chunking."""
from typing import Dict, Any, List, Optional
from app.ingestion.chunking.base import BaseChunker, Chunk

class FixedSizeChunker(BaseChunker):
    """Chunks text into deterministic sliding windows with configurable overlap."""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
            
        base_meta = metadata.copy() if metadata else {}
        step = max(1, self.chunk_size - self.chunk_overlap)
        chunks = []
        idx = 0
        
        for start in range(0, len(text), step):
            end = min(start + self.chunk_size, len(text))
            chunk_str = text[start:end].strip()
            if chunk_str:
                chunks.append(Chunk(
                    text=chunk_str,
                    metadata={**base_meta, "chunk_strategy": "fixed_size", "char_start": start, "char_end": end},
                    chunk_index=idx,
                    token_count=len(chunk_str.split())
                ))
                idx += 1
            if end >= len(text):
                break
                
        return chunks
