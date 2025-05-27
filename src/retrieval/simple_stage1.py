import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from pathlib import Path
import json
import time
import re

from indexing.multi_index import MultiIndex, build_index_from_csv

logger = logging.getLogger(__name__)

class SimplePreprocessor:
    """Simplified preprocessor without external dependencies."""
    
    def preprocess(self, text: str) -> Dict:
        """Simple text preprocessing."""
        # Basic cleaning
        cleaned = re.sub(r'[।॥]+', ' ', text)  # Remove dandas
        cleaned = re.sub(r'[0-9]+', '', cleaned)  # Remove numbers
        cleaned = re.sub(r'[/\[\](){}]+', ' ', cleaned)  # Remove brackets
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # Normalize whitespace
        
        # Convert to lowercase for ASCII version
        ascii_form = cleaned.lower()
        
        # Simple word splitting
        words = cleaned.split()
        
        return {
            'original': text,
            'cleaned': cleaned,
            'ascii': ascii_form,
            'words': words,
            'lemmas_text': cleaned  # Use cleaned text as lemmas
        }

class Stage1Retriever:
    """
    Stage 1: Coarse retrieval system for Sanskrit quotes.
    Goal: Recall@100 ≥ 95%
    """
    
    def __init__(self, index_path: Optional[str] = None):
        self.multi_index = MultiIndex()
        self.preprocessor = SimplePreprocessor()
        self.index_path = index_path
        self.is_loaded = False
        
        if index_path and Path(index_path).exists():
            self.load_index(index_path)
    
    def build_index(self, csv_path: str, text_column: str = 'text', save_path: Optional[str] = None):
        """Build the retrieval index from a CSV file."""
        logger.info(f"Building index from {csv_path}")
        
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded {len(df)} documents")
        
        # Preprocess documents
        processed_docs = []
        metadata = []
        
        for idx, row in df.iterrows():
            text = str(row[text_column]) if pd.notna(row[text_column]) else ""
            
            if len(text.strip()) > 10:  # Skip very short texts
                # Preprocess the text
                processed = self.preprocessor.preprocess(text)
                
                # Use cleaned text for indexing
                processed_text = processed['lemmas_text']
                
                processed_docs.append(processed_text)
                
                # Store metadata
                meta = row.to_dict()
                meta['original_text'] = text
                meta['processed_text'] = processed_text
                meta['doc_index'] = len(processed_docs) - 1
                metadata.append(meta)
        
        logger.info(f"Preprocessed {len(processed_docs)} valid documents")
        
        # Build the multi-index
        self.multi_index.fit(processed_docs, metadata)
        self.is_loaded = True
        
        # Save if path provided
        if save_path:
            self.save_index(save_path)
        
        return len(processed_docs)
    
    def load_index(self, path: str):
        """Load pre-built index."""
        self.multi_index.load(path)
        self.index_path = path
        self.is_loaded = True
        logger.info(f"Index loaded from {path}")
    
    def save_index(self, path: str):
        """Save the index."""
        self.multi_index.save(path)
        logger.info(f"Index saved to {path}")
    
    def retrieve(self, query: str, top_k: int = 100, preprocess_query: bool = True) -> List[Dict]:
        """
        Retrieve top-k candidates for a query.
        
        Args:
            query: Input Sanskrit quote
            top_k: Number of candidates to retrieve
            preprocess_query: Whether to preprocess the query
            
        Returns:
            List of candidate documents with scores and metadata
        """
        if not self.is_loaded:
            raise ValueError("Index not loaded. Call build_index() or load_index() first.")
        
        start_time = time.time()
        
        # Preprocess query if requested
        if preprocess_query:
            processed_query = self.preprocessor.preprocess(query)
            search_query = processed_query['lemmas_text']
        else:
            search_query = query
        
        # Search using multi-index
        results = self.multi_index.search(search_query, top_k=top_k, use_synonyms=True)
        
        # Add timing information
        retrieval_time = time.time() - start_time
        
        # Enhance results with additional information
        enhanced_results = []
        for result in results:
            enhanced_result = {
                'rank': len(enhanced_results) + 1,
                'score': result['score'],
                'text': result['metadata']['original_text'],
                'processed_text': result['metadata']['processed_text'],
                'source_file': result['metadata'].get('source_file', ''),
                'work': result['metadata'].get('work', ''),
                'category': result['metadata'].get('category', ''),
                'quote_id': result['metadata'].get('quote_id', ''),
                'doc_index': result['metadata'].get('doc_index', result['doc_id']),
                'retrieval_time': retrieval_time
            }
            enhanced_results.append(enhanced_result)
        
        logger.info(f"Retrieved {len(enhanced_results)} candidates in {retrieval_time:.3f}s")
        return enhanced_results
    
    def batch_retrieve(self, queries: List[str], top_k: int = 100) -> List[List[Dict]]:
        """Retrieve candidates for multiple queries."""
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            query_results = self.retrieve(query, top_k)
            results.append(query_results)
        
        return results

class RetrievalEvaluator:
    """Evaluate the performance of Stage 1 retrieval."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate_recall(self, retriever: Stage1Retriever, test_queries: List[Dict], 
                       k_values: List[int] = [1, 5, 10, 20, 50, 100]) -> Dict:
        """
        Evaluate recall@k for test queries.
        
        Args:
            retriever: The retrieval system
            test_queries: List of test queries with ground truth
            k_values: List of k values to evaluate
            
        Returns:
            Dictionary with recall metrics
        """
        recall_scores = {f'recall@{k}': [] for k in k_values}
        
        for query_data in test_queries:
            query = query_data['query']
            ground_truth_id = query_data.get('ground_truth_id', query_data.get('quote_id'))
            
            # Retrieve candidates
            results = retriever.retrieve(query, top_k=max(k_values))
            
            # Check if ground truth is in top-k for each k
            retrieved_ids = [r.get('quote_id', r.get('doc_index')) for r in results]
            
            for k in k_values:
                is_found = ground_truth_id in retrieved_ids[:k]
                recall_scores[f'recall@{k}'].append(1.0 if is_found else 0.0)
        
        # Calculate mean recall scores
        mean_recall = {}
        for metric, scores in recall_scores.items():
            mean_recall[metric] = np.mean(scores) if scores else 0.0
        
        self.metrics.update(mean_recall)
        return mean_recall
    
    def create_test_queries(self, df: pd.DataFrame, num_queries: int = 50) -> List[Dict]:
        """Create test queries by sampling from the dataset."""
        # Sample random documents
        sampled = df.sample(n=min(num_queries, len(df)), random_state=42)
        
        test_queries = []
        for _, row in sampled.iterrows():
            # Create query by taking a substring of the text
            text = str(row['text'])
            
            # Take roughly half the text as query
            words = text.split()
            if len(words) > 4:
                query_words = words[:len(words)//2]
                query = ' '.join(query_words)
            else:
                query = text
            
            test_query = {
                'query': query,
                'ground_truth_id': row.get('quote_id', row.name),
                'ground_truth_text': text,
                'work': row.get('work', ''),
                'category': row.get('category', '')
            }
            test_queries.append(test_query)
        
        return test_queries
    
    def run_evaluation(self, retriever: Stage1Retriever, csv_path: str, 
                      num_test_queries: int = 50) -> Dict:
        """Run complete evaluation."""
        logger.info("Running Stage 1 retrieval evaluation...")
        
        # Load dataset
        df = pd.read_csv(csv_path)
        
        # Create test queries
        test_queries = self.create_test_queries(df, num_test_queries)
        logger.info(f"Created {len(test_queries)} test queries")
        
        # Evaluate recall
        recall_metrics = self.evaluate_recall(retriever, test_queries)
        
        # Log results
        logger.info("Recall Results:")
        for metric, score in recall_metrics.items():
            logger.info(f"  {metric}: {score:.3f}")
        
        return {
            'recall_metrics': recall_metrics,
            'num_test_queries': len(test_queries),
            'test_queries': test_queries
        }

def main():
    """Test Stage 1 retrieval system."""
    logging.basicConfig(level=logging.INFO)
    
    # Paths
    csv_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    index_path = "/home/sai/Desktop/ullu/data/stage1_index.pkl"
    
    # Initialize retriever
    retriever = Stage1Retriever()
    
    # Build index
    logger.info("Building Stage 1 retrieval index...")
    retriever.build_index(csv_path, save_path=index_path)
    
    # Test some queries
    test_queries = [
        "dharma",
        "viṣṇu hari",
        "devī cakre bhayākulā", 
        "namaḥ svadhāyai",
        "rājasya dharma"
    ]
    
    print("\n" + "="*60)
    print("STAGE 1 RETRIEVAL TESTING")
    print("="*60)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 40)
        
        results = retriever.retrieve(query, top_k=5)
        
        for result in results:
            print(f"Rank {result['rank']}: Score={result['score']:.3f}")
            print(f"  Text: {result['text'][:80]}...")
            print(f"  Work: {result['work']} ({result['category']})")
    
    # Run evaluation
    evaluator = RetrievalEvaluator()
    eval_results = evaluator.run_evaluation(retriever, csv_path, num_test_queries=20)
    
    print(f"\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for metric, score in eval_results['recall_metrics'].items():
        print(f"{metric}: {score:.1%}")

if __name__ == "__main__":
    main()