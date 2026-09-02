"""Recursive character splitting strategy using hierarchical separators."""
from typing import Dict, Any, List, Optional
from app.ingestion.chunking.base import BaseChunker, Chunk

class RecursiveChunker(BaseChunker):
    """Hierarchically splits text across markdown headers, paragraphs, sentences, and words."""
    
    SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
            
        base_meta = metadata.copy() if metadata else {}
        split_texts = self._split_text(text, self.SEPARATORS)
        
        chunks = []
        for idx, item in enumerate(split_texts):
            clean_item = item.strip()
            if clean_item:
                chunks.append(Chunk(
                    text=clean_item,
                    metadata={**base_meta, "chunk_strategy": "recursive"},
                    chunk_index=idx,
                    token_count=len(clean_item.split())
                ))
        return chunks

    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        final_chunks = []
        separator = separators[-1]
        new_separators = []
        
        for i, _s in enumerate(separators):
            if _s == "":
                separator = _s
                break
            if _s in text:
                separator = _s
                new_separators = separators[i + 1:]
                break

        splits = text.split(separator) if separator else list(text)
        good_splits = []
        
        for s in splits:
            if not s:
                continue
            if len(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []
                if not new_separators:
                    final_chunks.append(s[:self.chunk_size])
                else:
                    other_info = self._split_text(s, new_separators)
                    final_chunks.extend(other_info)
                    
        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)
            
        return final_chunks

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        docs = []
        current_doc = []
        total = 0
        
        for d in splits:
            _len = len(d)
            if total + _len + (len(separator) if len(current_doc) > 0 else 0) > self.chunk_size:
                if total > 0:
                    doc = separator.join(current_doc)
                    if doc.strip():
                        docs.append(doc)
                    # Handle overlap
                    while total > self.chunk_overlap and current_doc:
                        popped = current_doc.pop(0)
                        total -= len(popped) + len(separator)
                current_doc.append(d)
                total += _len
            else:
                current_doc.append(d)
                total += _len + (len(separator) if len(current_doc) > 1 else 0)
                
        if current_doc:
            doc = separator.join(current_doc)
            if doc.strip():
                docs.append(doc)
                
        return docs
