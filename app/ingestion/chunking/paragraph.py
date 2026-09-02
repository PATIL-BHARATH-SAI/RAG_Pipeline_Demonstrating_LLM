"""Paragraph-based chunking strategy."""
from typing import Dict, Any, List, Optional
from app.ingestion.chunking.base import BaseChunker, Chunk

class ParagraphChunker(BaseChunker):
    """Chunks text preserving natural paragraph boundaries with max length cap."""
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
            
        base_meta = metadata.copy() if metadata else {}
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        
        chunks = []
        current_chunk_text = []
        current_length = 0
        idx = 0
        
        for para in paragraphs:
            para_len = len(para)
            if current_length + para_len > self.chunk_size and current_chunk_text:
                full_text = "\n\n".join(current_chunk_text)
                chunks.append(Chunk(
                    text=full_text,
                    metadata={**base_meta, "chunk_strategy": "paragraph"},
                    chunk_index=idx,
                    token_count=len(full_text.split())
                ))
                idx += 1
                current_chunk_text = [para]
                current_length = para_len
            else:
                current_chunk_text.append(para)
                current_length += para_len + 2
                
        if current_chunk_text:
            full_text = "\n\n".join(current_chunk_text)
            chunks.append(Chunk(
                text=full_text,
                metadata={**base_meta, "chunk_strategy": "paragraph"},
                chunk_index=idx,
                token_count=len(full_text.split())
            ))
            
        return chunks
