"""Semantic breakpoint chunking strategy based on embedding cosine similarity transitions."""
import re
import numpy as np
from typing import Dict, Any, List, Optional
from app.ingestion.chunking.base import BaseChunker, Chunk

class SemanticChunker(BaseChunker):
    """Splits text into sentences, computes adjacency similarity, and splits at semantic valleys."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 150, similarity_threshold: float = 0.75, embedding_provider = None):
        super().__init__(chunk_size, chunk_overlap)
        self.similarity_threshold = similarity_threshold
        self.embedding_provider = embedding_provider

    def _split_into_sentences(self, text: str) -> List[str]:
        # Split by punctuation while keeping natural boundaries
        sentences = re.split(r'(?<=[.?!])\s+', text)
        return [s.strip() for s in sentences if s.strip()]

    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        if not text or not text.strip():
            return []
            
        base_meta = metadata.copy() if metadata else {}
        sentences = self._split_into_sentences(text)
        if len(sentences) <= 1:
            return [Chunk(text=text.strip(), metadata={**base_meta, "chunk_strategy": "semantic"}, chunk_index=0)]

        # If embedding provider is passed, compute embeddings for semantic boundaries
        if self.embedding_provider:
            try:
                embeddings = self.embedding_provider.embed_texts(sentences)
                # Normalize embeddings
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1e-10
                norm_embeds = embeddings / norms
                
                # Compute adjacent similarities
                similarities = np.sum(norm_embeds[:-1] * norm_embeds[1:], axis=1)
                
                # Identify split points where similarity drops below threshold
                splits = [0]
                current_len = 0
                for i, sim in enumerate(similarities):
                    current_len += len(sentences[i])
                    if sim < self.similarity_threshold or current_len >= self.chunk_size:
                        splits.append(i + 1)
                        current_len = 0
                splits.append(len(sentences))
                splits = sorted(list(set(splits)))
                
                chunks = []
                for idx in range(len(splits) - 1):
                    chunk_sent = sentences[splits[idx]:splits[idx+1]]
                    chunk_text = " ".join(chunk_sent).strip()
                    if chunk_text:
                        chunks.append(Chunk(
                            text=chunk_text,
                            metadata={**base_meta, "chunk_strategy": "semantic"},
                            chunk_index=idx,
                            token_count=len(chunk_text.split())
                        ))
                return chunks
            except Exception:
                pass # Fallback to sentence group chunking
                
        # Fast rule-based semantic sentence grouping fallback
        chunks = []
        current_sentences = []
        current_len = 0
        idx = 0
        for sent in sentences:
            if current_len + len(sent) > self.chunk_size and current_sentences:
                chunk_str = " ".join(current_sentences)
                chunks.append(Chunk(
                    text=chunk_str,
                    metadata={**base_meta, "chunk_strategy": "semantic_rule"},
                    chunk_index=idx,
                    token_count=len(chunk_str.split())
                ))
                idx += 1
                current_sentences = [sent]
                current_len = len(sent)
            else:
                current_sentences.append(sent)
                current_len += len(sent) + 1
                
        if current_sentences:
            chunk_str = " ".join(current_sentences)
            chunks.append(Chunk(
                text=chunk_str,
                metadata={**base_meta, "chunk_strategy": "semantic_rule"},
                chunk_index=idx,
                token_count=len(chunk_str.split())
            ))
            
        return chunks
