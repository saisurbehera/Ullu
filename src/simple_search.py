"""
Simple fallback search functionality when full multi-index is not available.
Provides basic text search using built-in Python capabilities.
"""

import pandas as pd
import re
from typing import List, Tuple
from collections import Counter

class SimpleSearch:
    """Simple search implementation using basic text matching."""
    
    def __init__(self):
        self.passages = []
        self.indexed = False
    
    def build_index(self, passages: List[str]):
        """Build simple index from passages."""
        self.passages = passages
        self.indexed = True
        print(f"Simple search index built with {len(passages)} passages")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """Simple search using text matching."""
        if not self.indexed:
            return []
        
        query_terms = query.lower().split()
        results = []
        
        for idx, passage in enumerate(self.passages):
            passage_lower = passage.lower()
            score = self._calculate_score(query_terms, passage_lower)
            
            if score > 0:
                results.append((idx, score))
        
        # Sort by score descending
        results.sort(key=lambda x: x[1], reverse=True)
        
        return results[:top_k]
    
    def _calculate_score(self, query_terms: List[str], passage: str) -> float:
        """Calculate simple relevance score."""
        score = 0.0
        passage_words = passage.split()
        
        for term in query_terms:
            # Exact word match
            if term in passage_words:
                score += 2.0
            
            # Partial match
            elif any(term in word for word in passage_words):
                score += 1.0
            
            # Substring match
            elif term in passage:
                score += 0.5
        
        # Boost score for multiple term matches
        if len(query_terms) > 1:
            matched_terms = sum(1 for term in query_terms if term in passage)
            if matched_terms > 1:
                score *= (1 + matched_terms * 0.2)
        
        # Normalize by passage length (prevent very long passages from dominating)
        if len(passage_words) > 0:
            score = score / (1 + len(passage_words) / 100)
        
        return score