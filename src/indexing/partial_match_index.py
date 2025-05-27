"""
Partial text matching index for Sanskrit quote retrieval.
Handles substring and partial quote matching.
"""

import re
from typing import List, Dict, Tuple, Set
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class PartialMatchIndex:
    """Index optimized for partial text matching."""
    
    def __init__(self, window_size: int = 10, overlap: int = 5):
        """
        Initialize partial match index.
        
        Args:
            window_size: Number of words in each indexed window
            overlap: Number of overlapping words between windows
        """
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = window_size - overlap
        
        # Storage
        self.documents = []
        self.windows = []  # List of (doc_id, start_pos, end_pos, text)
        self.window_to_docs = defaultdict(set)  # window_text -> set of (doc_id, pos)
        self.inverted_index = defaultdict(set)  # word -> set of window_ids
        
    def fit(self, documents: List[str]):
        """Build partial match index from documents."""
        self.documents = documents
        self.windows = []
        self.window_to_docs.clear()
        self.inverted_index.clear()
        
        logger.info(f"Building partial match index for {len(documents)} documents")
        
        for doc_id, doc in enumerate(tqdm(documents, desc="Indexing documents")):
            # Tokenize document
            words = doc.split()
            
            # Create overlapping windows
            for start in range(0, len(words), self.step_size):
                end = min(start + self.window_size, len(words))
                window_words = words[start:end]
                window_text = ' '.join(window_words)
                
                # Store window
                window_id = len(self.windows)
                self.windows.append({
                    'id': window_id,
                    'doc_id': doc_id,
                    'start': start,
                    'end': end,
                    'text': window_text,
                    'words': window_words
                })
                
                # Update inverted index
                for word in set(window_words):
                    self.inverted_index[word].add(window_id)
                
                # Map window to document
                self.window_to_docs[window_text].add((doc_id, start))
        
        logger.info(f"Created {len(self.windows)} windows from {len(documents)} documents")
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """
        Search for documents containing the query as a substring.
        
        Returns:
            List of (doc_id, score) tuples
        """
        query_words = query.split()
        query_len = len(query_words)
        
        if query_len == 0:
            return []
        
        # Find candidate windows containing query words
        candidate_windows = set()
        
        # Start with windows containing the first query word
        if query_words[0] in self.inverted_index:
            candidate_windows = self.inverted_index[query_words[0]].copy()
        
        # Intersect with windows containing other query words
        for word in query_words[1:]:
            if word in self.inverted_index:
                candidate_windows &= self.inverted_index[word]
        
        # Score each candidate window
        doc_scores = defaultdict(float)
        doc_matches = defaultdict(list)
        
        for window_id in candidate_windows:
            window = self.windows[window_id]
            window_words = window['words']
            doc_id = window['doc_id']
            
            # Check for exact substring match
            score, match_info = self._score_window_match(query_words, window_words)
            
            if score > 0:
                doc_scores[doc_id] = max(doc_scores[doc_id], score)
                doc_matches[doc_id].append({
                    'score': score,
                    'window_id': window_id,
                    'match_info': match_info
                })
        
        # Also check for cross-window matches
        self._check_cross_window_matches(query_words, doc_scores, doc_matches)
        
        # Sort by score
        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        
        return sorted_docs[:top_k]
    
    def _score_window_match(self, query_words: List[str], window_words: List[str]) -> Tuple[float, Dict]:
        """Score how well query matches within a window."""
        query_len = len(query_words)
        window_len = len(window_words)
        
        if query_len > window_len:
            return 0.0, {}
        
        best_score = 0.0
        best_match = {}
        
        # Check all possible positions
        for start in range(window_len - query_len + 1):
            # Check exact match
            if window_words[start:start + query_len] == query_words:
                return 1.0, {
                    'type': 'exact',
                    'position': start,
                    'length': query_len
                }
            
            # Calculate partial match score
            matches = sum(1 for i in range(query_len) 
                         if window_words[start + i] == query_words[i])
            score = matches / query_len
            
            if score > best_score:
                best_score = score
                best_match = {
                    'type': 'partial',
                    'position': start,
                    'matches': matches,
                    'length': query_len
                }
        
        return best_score, best_match
    
    def _check_cross_window_matches(self, query_words: List[str], 
                                   doc_scores: Dict[int, float], 
                                   doc_matches: Dict[int, List]):
        """Check for matches that span across windows."""
        query_len = len(query_words)
        
        # Group windows by document
        doc_windows = defaultdict(list)
        for window in self.windows:
            doc_windows[window['doc_id']].append(window)
        
        # Check each document
        for doc_id, windows in doc_windows.items():
            # Sort windows by position
            sorted_windows = sorted(windows, key=lambda w: w['start'])
            
            # Check adjacent windows
            for i in range(len(sorted_windows) - 1):
                w1 = sorted_windows[i]
                w2 = sorted_windows[i + 1]
                
                # Get overlapping region
                overlap_start = w2['start']
                overlap_end = w1['end']
                
                if overlap_start < overlap_end:
                    # Combine words from both windows
                    combined_words = w1['words'] + w2['words'][overlap_end - overlap_start:]
                    
                    # Check for match in combined region
                    score, match_info = self._score_window_match(query_words, combined_words)
                    
                    if score > doc_scores[doc_id]:
                        doc_scores[doc_id] = score
                        match_info['type'] = 'cross_window'
                        match_info['windows'] = [w1['id'], w2['id']]
                        doc_matches[doc_id].append({
                            'score': score,
                            'match_info': match_info
                        })
    
    def get_match_context(self, doc_id: int, position: int, context_size: int = 5) -> str:
        """Get context around a match position."""
        if doc_id >= len(self.documents):
            return ""
        
        doc_words = self.documents[doc_id].split()
        start = max(0, position - context_size)
        end = min(len(doc_words), position + context_size)
        
        return ' '.join(doc_words[start:end])
    
    def extract_all_substrings(self, min_length: int = 3, max_length: int = 20) -> Dict[str, Set[int]]:
        """
        Extract all substrings of given length range for exact matching.
        Memory intensive - use only for small corpora.
        """
        substring_index = defaultdict(set)
        
        for doc_id, doc in enumerate(tqdm(self.documents, desc="Extracting substrings")):
            words = doc.split()
            
            for length in range(min_length, min(max_length + 1, len(words) + 1)):
                for start in range(len(words) - length + 1):
                    substring = ' '.join(words[start:start + length])
                    substring_index[substring].add(doc_id)
        
        return substring_index


class EnhancedMultiIndex:
    """Enhanced multi-index with partial matching support."""
    
    def __init__(self, base_index=None):
        self.base_index = base_index
        self.partial_index = PartialMatchIndex()
        self.is_fitted = False
        
    def fit(self, documents: List[str], metadata: List[Dict] = None):
        """Fit both base and partial indices."""
        # Fit base index if provided
        if self.base_index:
            self.base_index.fit(documents, metadata)
        
        # Fit partial match index
        self.partial_index.fit(documents)
        
        self.documents = documents
        self.metadata = metadata or [{'id': i} for i in range(len(documents))]
        self.is_fitted = True
        
    def search(self, query: str, top_k: int = 100, use_partial: bool = True) -> List[Dict]:
        """
        Search with both full and partial matching.
        
        Args:
            query: Search query
            top_k: Number of results to return
            use_partial: Whether to use partial matching
        """
        if not self.is_fitted:
            raise ValueError("Index not fitted. Call fit() first.")
        
        all_results = {}
        
        # Get results from base index
        if self.base_index:
            base_results = self.base_index.search(query, top_k=top_k * 2)
            for result in base_results:
                doc_id = result['doc_id']
                all_results[doc_id] = result
        
        # Get results from partial index
        if use_partial:
            partial_results = self.partial_index.search(query, top_k=top_k * 2)
            
            for doc_id, score in partial_results:
                if doc_id in all_results:
                    # Combine scores (boost if found in both)
                    all_results[doc_id]['score'] = all_results[doc_id]['score'] * 0.7 + score * 0.5
                    all_results[doc_id]['partial_match'] = True
                else:
                    # Add new result from partial matching
                    all_results[doc_id] = {
                        'doc_id': doc_id,
                        'text': self.documents[doc_id],
                        'score': score * 0.8,  # Slightly lower weight for partial-only matches
                        'metadata': self.metadata[doc_id],
                        'partial_match': True
                    }
        
        # Sort by combined score
        sorted_results = sorted(all_results.values(), 
                               key=lambda x: x['score'], 
                               reverse=True)
        
        return sorted_results[:top_k]