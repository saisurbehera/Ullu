import json
import pickle
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from collections import defaultdict
import re
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Try to import KG components
try:
    from knowledge_graph import KGEnhancer
    kg_available = True
except ImportError:
    kg_available = False
    logger.info("Knowledge Graph enhancement not available")

# Try to import partial matching
try:
    from .partial_match_index import PartialMatchIndex
    partial_matching_available = True
except ImportError:
    partial_matching_available = False
    logger.info("Partial matching not available")

class BM25Index:
    """BM25 indexing for lexical retrieval."""
    
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.documents = []
        self.doc_freqs = defaultdict(int)
        self.idf = {}
        self.doc_len = []
        self.avgdl = 0
        self.N = 0
        
    def fit(self, documents: List[str]):
        """Build BM25 index from documents."""
        self.documents = documents
        self.N = len(documents)
        
        # Calculate document frequencies and lengths
        for doc in tqdm(documents, desc="Building BM25 index"):
            terms = doc.split()
            self.doc_len.append(len(terms))
            unique_terms = set(terms)
            
            for term in unique_terms:
                self.doc_freqs[term] += 1
        
        self.avgdl = sum(self.doc_len) / self.N
        
        # Calculate IDF
        for term, freq in self.doc_freqs.items():
            self.idf[term] = np.log((self.N - freq + 0.5) / (freq + 0.5))
    
    def score(self, query: str, doc_idx: int) -> float:
        """Calculate BM25 score for a query-document pair."""
        query_terms = query.split()
        doc_terms = self.documents[doc_idx].split()
        doc_len = self.doc_len[doc_idx]
        
        score = 0.0
        term_freqs = defaultdict(int)
        
        for term in doc_terms:
            term_freqs[term] += 1
        
        for term in query_terms:
            if term in term_freqs:
                tf = term_freqs[term]
                idf = self.idf.get(term, 0)
                
                score += idf * (tf * (self.k1 + 1)) / (
                    tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                )
        
        return score
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Search for top-k documents."""
        scores = []
        
        for idx in range(self.N):
            score = self.score(query, idx)
            scores.append((idx, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

class NGramIndex:
    """Character n-gram index for fuzzy matching."""
    
    def __init__(self, n: int = 3):
        self.n = n
        self.documents = []
        self.ngram_index = defaultdict(set)
        self.doc_ngrams = []
    
    def _extract_ngrams(self, text: str) -> set:
        """Extract character n-grams from text."""
        text = text.lower().replace(' ', '')
        ngrams = set()
        
        for i in range(len(text) - self.n + 1):
            ngrams.add(text[i:i + self.n])
        
        return ngrams
    
    def fit(self, documents: List[str]):
        """Build n-gram index."""
        self.documents = documents
        
        for idx, doc in enumerate(tqdm(documents, desc="Building N-gram index")):
            ngrams = self._extract_ngrams(doc)
            self.doc_ngrams.append(ngrams)
            
            for ngram in ngrams:
                self.ngram_index[ngram].add(idx)
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Search using n-gram overlap."""
        query_ngrams = self._extract_ngrams(query)
        
        if not query_ngrams:
            return []
        
        # Count n-gram overlaps
        doc_scores = defaultdict(int)
        
        for ngram in query_ngrams:
            for doc_idx in self.ngram_index.get(ngram, []):
                doc_scores[doc_idx] += 1
        
        # Calculate Jaccard similarity
        scored_docs = []
        for doc_idx, overlap_count in doc_scores.items():
            doc_ngrams = self.doc_ngrams[doc_idx]
            union_size = len(query_ngrams.union(doc_ngrams))
            
            if union_size > 0:
                jaccard = overlap_count / union_size
                scored_docs.append((doc_idx, jaccard))
        
        # Sort by score descending
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        return scored_docs[:top_k]

class DenseEmbeddingIndex:
    """Dense embedding index using sentence transformers."""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.embeddings = None
        self.documents = []
        
        try:
            from sentence_transformers import SentenceTransformer
            import torch
            
            # Check for GPU availability
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.model = SentenceTransformer(model_name, device=device)
            
            if device == 'cuda':
                gpu_name = torch.cuda.get_device_name(0)
                print(f"🚀 Loaded sentence transformer: {model_name} on GPU ({gpu_name})")
            else:
                print(f"✅ Loaded sentence transformer: {model_name} on CPU")
                
        except ImportError:
            logger.warning("sentence-transformers not available, using TF-IDF fallback")
            self.model = None
            self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
    
    def fit(self, documents: List[str]):
        """Build dense embedding index."""
        self.documents = documents
        
        if self.model is not None:
            # Use sentence transformers with GPU optimization
            print("🧠 Computing sentence embeddings on GPU...")
            batch_size = 1024  # Larger batch size for GPU
            self.embeddings = self.model.encode(
                documents, 
                convert_to_tensor=False, 
                show_progress_bar=True,
                batch_size=batch_size
            )
        else:
            # Fallback to TF-IDF
            print("📊 Computing TF-IDF embeddings...")
            self.embeddings = self.tfidf.fit_transform(tqdm(documents, desc="TF-IDF vectorization")).toarray()
    
    def search(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Search using dense embeddings."""
        if self.embeddings is None:
            return []
        
        if self.model is not None:
            query_embedding = self.model.encode([query], convert_to_tensor=False)
        else:
            query_embedding = self.tfidf.transform([query]).toarray()
        
        # Calculate cosine similarities
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k results
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = [(int(idx), float(similarities[idx])) for idx in top_indices]
        
        return results

class SynonymExpander:
    """Expand queries with Sanskrit synonyms and epithets."""
    
    def __init__(self):
        self.synonym_dict = {
            # Deity epithets
            'viṣṇu': ['hari', 'nārāyaṇa', 'keśava', 'madhusūdana', 'govinda'],
            'śiva': ['maheśvara', 'śaṅkara', 'rudra', 'īśa', 'hara'],
            'brahmā': ['pitāmaha', 'svayambhū', 'prajāpati', 'virañci'],
            'devī': ['ambā', 'umā', 'pārvatī', 'śakti', 'bhagavatī'],
            
            # Common synonyms
            'rāja': ['nṛpa', 'bhūpati', 'bhūpāla', 'mahīpati'],
            'veda': ['śruti', 'nigama', 'āgama'],
            'dharma': ['nyāya', 'rita', 'satya'],
            'artha': ['vitta', 'dhana', 'sampad'],
            'kāma': ['icchā', 'vāsanā', 'abhilāṣa'],
            'mokṣa': ['mukti', 'nirvāṇa', 'kaivalya', 'apavarga']
        }
    
    def expand_query(self, query: str) -> List[str]:
        """Expand query with synonyms."""
        expanded_queries = [query]
        query_lower = query.lower()
        
        for term, synonyms in self.synonym_dict.items():
            if term in query_lower:
                for synonym in synonyms:
                    expanded_query = query_lower.replace(term, synonym)
                    expanded_queries.append(expanded_query)
        
        return list(set(expanded_queries))

class MultiIndex:
    """Combined multi-index system for Sanskrit quote retrieval."""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Initialize component indices
        self.bm25_index = BM25Index()
        self.ngram_index = NGramIndex(n=3)
        self.dense_index = DenseEmbeddingIndex()
        self.synonym_expander = SynonymExpander()
        
        # Initialize KG enhancer if available
        self.kg_enhancer = None
        if kg_available and self.config.get('use_kg', True):
            try:
                self.kg_enhancer = KGEnhancer()
                logger.info("Knowledge Graph enhancement enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize KG enhancer: {e}")
        
        # Initialize partial matching if available
        self.partial_index = None
        if partial_matching_available and self.config.get('use_partial', True):
            try:
                self.partial_index = PartialMatchIndex(
                    window_size=self.config.get('window_size', 10),
                    overlap=self.config.get('overlap', 5)
                )
                logger.info("Partial matching enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize partial matching: {e}")
        
        # Weights for combining scores
        self.weights = {
            'bm25': 0.5,
            'ngram': 0.3,
            'dense': 0.2
        }
        
        self.documents = []
        self.metadata = []
        self.is_fitted = False
    
    def fit(self, documents: List[str], metadata: List[Dict] = None):
        """Build all indices."""
        logger.info(f"Building multi-index for {len(documents)} documents")
        
        self.documents = documents
        self.metadata = metadata or [{'id': i} for i in range(len(documents))]
        
        # Build component indices
        logger.info("Building BM25 index...")
        self.bm25_index.fit(documents)
        
        logger.info("Building n-gram index...")
        self.ngram_index.fit(documents)
        
        logger.info("Building dense embedding index...")
        self.dense_index.fit(documents)
        
        # Build partial matching index if enabled
        if self.partial_index:
            logger.info("Building partial matching index...")
            self.partial_index.fit(documents)
        
        self.is_fitted = True
        logger.info("Multi-index built successfully")
    
    def search(self, query: str, top_k: int = 100, use_synonyms: bool = True, use_kg: bool = True, use_partial: bool = True) -> List[Dict]:
        """Search across all indices and combine results."""
        if not self.is_fitted:
            raise ValueError("Index not fitted. Call fit() first.")
        
        # Expand query with synonyms if enabled
        queries = [query]
        if use_synonyms:
            queries = self.synonym_expander.expand_query(query)
        
        # Expand query with KG if enabled
        if use_kg and self.kg_enhancer:
            try:
                kg_expanded = self.kg_enhancer.expand_query(query, max_expansions=3)
                queries.extend(kg_expanded[1:])  # Skip first as it's the original
                queries = list(set(queries))  # Remove duplicates
            except Exception as e:
                logger.warning(f"KG query expansion failed: {e}")
        
        all_results = defaultdict(float)
        result_details = defaultdict(dict)  # Store additional info about results
        
        for q in queries:
            # Get results from each index
            bm25_results = self.bm25_index.search(q, top_k * 2)
            ngram_results = self.ngram_index.search(q, top_k * 2)
            dense_results = self.dense_index.search(q, top_k * 2)
            
            # Normalize and combine scores
            self._combine_results(all_results, bm25_results, 'bm25', len(queries))
            self._combine_results(all_results, ngram_results, 'ngram', len(queries))
            self._combine_results(all_results, dense_results, 'dense', len(queries))
        
        # Add partial matching results if enabled
        if use_partial and self.partial_index:
            try:
                partial_results = self.partial_index.search(query, top_k * 2)
                
                for doc_idx, score in partial_results:
                    # Boost existing results or add new ones
                    if doc_idx in all_results:
                        # Document found by both methods - strong signal
                        all_results[doc_idx] = all_results[doc_idx] * 0.7 + score * 0.5
                    else:
                        # Found only by partial matching
                        all_results[doc_idx] = score * 0.8
                    
                    result_details[doc_idx]['partial_match'] = True
                    result_details[doc_idx]['partial_score'] = score
                    
            except Exception as e:
                logger.warning(f"Partial matching failed: {e}")
        
        # Sort by combined score
        sorted_results = sorted(all_results.items(), key=lambda x: x[1], reverse=True)
        
        # Format results
        final_results = []
        for doc_idx, score in sorted_results[:top_k]:
            result = {
                'doc_id': doc_idx,
                'text': self.documents[doc_idx],
                'score': score,
                'metadata': self.metadata[doc_idx]
            }
            
            # Add partial match info if available
            if doc_idx in result_details:
                result.update(result_details[doc_idx])
            
            final_results.append(result)
        
        # Apply KG enhancement if enabled
        if use_kg and self.kg_enhancer and final_results:
            try:
                final_results = self.kg_enhancer.enhance_candidates(query, final_results[:top_k])
            except Exception as e:
                logger.warning(f"KG enhancement failed: {e}")
        
        return final_results[:top_k]
    
    def _combine_results(self, all_results: Dict, results: List[Tuple], 
                        index_type: str, num_queries: int):
        """Combine results from one index type."""
        weight = self.weights.get(index_type, 0.0)
        
        if not results:
            return
        
        # Normalize scores to [0, 1]
        max_score = max(score for _, score in results) if results else 1.0
        if max_score == 0:
            max_score = 1.0
        
        for doc_idx, score in results:
            normalized_score = score / max_score
            all_results[doc_idx] += (weight * normalized_score) / num_queries
    
    def save(self, path: str):
        """Save the multi-index to disk."""
        index_data = {
            'documents': self.documents,
            'metadata': self.metadata,
            'bm25_index': self.bm25_index,
            'ngram_index': self.ngram_index,
            'dense_index': self.dense_index,  # Save entire dense_index object
            'weights': self.weights,
            'config': self.config
        }
        
        # Save partial index if available
        if self.partial_index:
            index_data['partial_index'] = self.partial_index
        
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"Multi-index saved to {path}")
    
    def load(self, path: str):
        """Load the multi-index from disk."""
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.documents = index_data['documents']
        self.metadata = index_data['metadata']
        self.bm25_index = index_data['bm25_index']
        self.ngram_index = index_data['ngram_index']
        
        # Handle dense_index loading properly
        if 'dense_index' in index_data:
            self.dense_index = index_data['dense_index']
        else:
            # Legacy format - only embeddings saved
            self.dense_index.embeddings = index_data.get('dense_embeddings')
            # Reinitialize documents for the dense index
            self.dense_index.documents = self.documents
            
        self.weights = index_data['weights']
        self.config = index_data['config']
        
        # Load partial index if available
        if 'partial_index' in index_data and partial_matching_available:
            self.partial_index = index_data['partial_index']
        
        self.is_fitted = True
        
        logger.info(f"Multi-index loaded from {path}")

def build_index_from_csv(csv_path: str, text_column: str = 'text') -> MultiIndex:
    """Build multi-index from CSV file."""
    df = pd.read_csv(csv_path)
    
    documents = df[text_column].fillna('').tolist()
    metadata = df.to_dict('records')
    
    multi_index = MultiIndex()
    multi_index.fit(documents, metadata)
    
    return multi_index

def main():
    """Test the multi-index system."""
    # Build index from our Sanskrit quotes
    csv_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    
    logger.info("Building multi-index from Sanskrit quotes...")
    multi_index = build_index_from_csv(csv_path)
    
    # Test queries
    test_queries = [
        "dharma",
        "viṣṇu",
        "devī cakre",
        "rāja",
        "namaḥ"
    ]
    
    print("Testing multi-index search:")
    print("=" * 50)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        results = multi_index.search(query, top_k=3)
        
        for i, result in enumerate(results[:3], 1):
            print(f"{i}. Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            print(f"   Source: {result['metadata']['work']}")
    
    # Save index
    index_path = "/home/sai/Desktop/ullu/data/multi_index.pkl"
    multi_index.save(index_path)
    print(f"\nIndex saved to {index_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
