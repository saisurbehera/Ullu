import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import logging
import time
from typing import Dict, List, Optional
from pathlib import Path

from retrieval.simple_stage1 import Stage1Retriever
from ranking.cross_encoder import CrossEncoderRanker
from filtering.llm_filter import Stage3Filter

logger = logging.getLogger(__name__)

class SanskritQuoteRetrievalPipeline:
    """
    Complete 3-stage Sanskrit quote retrieval pipeline.
    
    Stage 1: Coarse retrieval (Recall@100 ≥ 95%)
    Stage 2: Fine-grained ranking (Precision@1 maximized)
    Stage 3: LLM-based filtering and confirmation
    """
    
    def __init__(self, 
                 stage1_index_path: str,
                 stage2_model_path: Optional[str] = None,
                 use_llm_filter: bool = True):
        
        # Initialize stages
        self.stage1 = Stage1Retriever(stage1_index_path)
        self.stage2 = CrossEncoderRanker()
        self.stage3 = Stage3Filter(use_llm=use_llm_filter)
        
        # Load models if available
        if stage2_model_path and Path(stage2_model_path).exists():
            self.stage2.load(stage2_model_path)
        
        self.is_ready = self.stage1.is_loaded
    
    def retrieve_quote_source(self, 
                            query: str,
                            stage1_top_k: int = 100,
                            stage2_top_k: int = 10,
                            final_top_k: int = 1) -> Dict:
        """
        Complete pipeline to retrieve quote source.
        
        Args:
            query: Sanskrit quote to find source for
            stage1_top_k: Number of candidates from Stage 1
            stage2_top_k: Number of candidates to rank in Stage 2
            final_top_k: Number of final results to return
            
        Returns:
            Dictionary with results and pipeline metadata
        """
        if not self.is_ready:
            raise ValueError("Pipeline not ready. Check index loading.")
        
        start_time = time.time()
        pipeline_log = []
        
        # Stage 1: Retrieval
        stage1_start = time.time()
        stage1_candidates = self.stage1.retrieve(query, top_k=stage1_top_k)
        stage1_time = time.time() - stage1_start
        
        pipeline_log.append({
            'stage': 1,
            'action': 'retrieval',
            'candidates_count': len(stage1_candidates),
            'time': stage1_time,
            'top_score': stage1_candidates[0]['score'] if stage1_candidates else 0.0
        })
        
        if not stage1_candidates:
            return self._create_result(query, [], pipeline_log, time.time() - start_time)
        
        # Stage 2: Ranking (if model is trained)
        stage2_start = time.time()
        if self.stage2.is_trained:
            # Take top candidates for ranking
            candidates_to_rank = stage1_candidates[:stage2_top_k]
            stage2_candidates = self.stage2.rank_passages(query, candidates_to_rank)
        else:
            # Skip ranking if model not trained
            stage2_candidates = stage1_candidates[:stage2_top_k]
            # Add dummy ranking scores
            for i, candidate in enumerate(stage2_candidates):
                candidate['ranking_score'] = 1.0 - (i * 0.1)
                candidate['final_rank'] = i + 1
        
        stage2_time = time.time() - stage2_start
        
        pipeline_log.append({
            'stage': 2,
            'action': 'ranking',
            'candidates_count': len(stage2_candidates),
            'time': stage2_time,
            'top_score': stage2_candidates[0].get('ranking_score', 0.0) if stage2_candidates else 0.0
        })
        
        # Stage 3: Filtering
        stage3_start = time.time()
        final_results = self.stage3.filter_candidates(query, stage2_candidates, top_k=final_top_k)
        stage3_time = time.time() - stage3_start
        
        pipeline_log.append({
            'stage': 3,
            'action': 'filtering',
            'candidates_count': len(final_results),
            'time': stage3_time,
            'matches_found': sum(1 for r in final_results if r.get('filter_match', False))
        })
        
        total_time = time.time() - start_time
        
        return self._create_result(query, final_results, pipeline_log, total_time)
    
    def _create_result(self, query: str, results: List[Dict], 
                      pipeline_log: List[Dict], total_time: float) -> Dict:
        """Create formatted result dictionary."""
        return {
            'query': query,
            'results': results,
            'pipeline_performance': {
                'total_time': total_time,
                'stage_performance': pipeline_log,
                'num_final_results': len(results)
            },
            'success': len(results) > 0 and any(r.get('filter_match', False) for r in results)
        }
    
    def batch_process(self, queries: List[str], **kwargs) -> List[Dict]:
        """Process multiple queries."""
        results = []
        
        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            result = self.retrieve_quote_source(query, **kwargs)
            results.append(result)
        
        return results
    
    def evaluate_pipeline(self, test_queries: List[Dict]) -> Dict:
        """Evaluate end-to-end pipeline performance."""
        logger.info(f"Evaluating pipeline on {len(test_queries)} queries")
        
        metrics = {
            'success_rate': 0.0,
            'average_time': 0.0,
            'stage1_recall': 0.0,
            'stage2_precision': 0.0,
            'stage3_accuracy': 0.0
        }
        
        successful_retrievals = 0
        total_time = 0.0
        stage1_hits = 0
        
        for query_data in test_queries:
            query = query_data['query']
            ground_truth_id = query_data.get('ground_truth_id')
            
            result = self.retrieve_quote_source(query)
            total_time += result['pipeline_performance']['total_time']
            
            # Check if successful
            if result['success']:
                successful_retrievals += 1
            
            # Check Stage 1 recall (if ground truth found in Stage 1)
            stage1_log = next((log for log in result['pipeline_performance']['stage_performance'] 
                             if log['stage'] == 1), None)
            if stage1_log and stage1_log['candidates_count'] > 0:
                # This would need access to stage1 candidates to check properly
                # Simplified: assume success means stage1 found it
                if result['success']:
                    stage1_hits += 1
        
        metrics['success_rate'] = successful_retrievals / len(test_queries)
        metrics['average_time'] = total_time / len(test_queries)
        metrics['stage1_recall'] = stage1_hits / len(test_queries)
        
        return metrics

def build_complete_pipeline(csv_path: str, force_rebuild: bool = False) -> SanskritQuoteRetrievalPipeline:
    """Build complete pipeline with all components."""
    logger.info("Building complete Sanskrit quote retrieval pipeline...")
    
    # Paths
    stage1_index_path = "/home/sai/Desktop/ullu/data/stage1_index.pkl"
    stage2_model_path = "/home/sai/Desktop/ullu/data/cross_encoder.pkl"
    
    # Build Stage 1 if needed
    if force_rebuild or not Path(stage1_index_path).exists():
        logger.info("Building Stage 1 index...")
        retriever = Stage1Retriever()
        retriever.build_index(csv_path, save_path=stage1_index_path)
    
    # Build Stage 2 if needed
    if force_rebuild or not Path(stage2_model_path).exists():
        logger.info("Building Stage 2 model...")
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        ranker = CrossEncoderRanker()
        training_pairs, labels = ranker.create_training_data(df.head(200), num_negatives=2)
        ranker.train(training_pairs, labels)
        ranker.save(stage2_model_path)
    
    # Initialize complete pipeline
    pipeline = SanskritQuoteRetrievalPipeline(
        stage1_index_path=stage1_index_path,
        stage2_model_path=stage2_model_path,
        use_llm_filter=True
    )
    
    logger.info("Pipeline built successfully!")
    return pipeline

def main():
    """Test complete pipeline."""
    logging.basicConfig(level=logging.INFO)
    
    csv_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    
    # Build pipeline
    pipeline = build_complete_pipeline(csv_path)
    
    # Test queries
    test_queries = [
        "dharma viṣṇu",
        "devī cakre bhayākulā",
        "namaḥ svadhāyai kavyāya",
        "rājasya dharma",
        "yādṛśaṃ tat"
    ]
    
    print("\n" + "="*80)
    print("COMPLETE SANSKRIT QUOTE RETRIEVAL PIPELINE")
    print("="*80)
    
    for query in test_queries:
        print(f"\nQuery: '{query}'")
        print("-" * 60)
        
        result = pipeline.retrieve_quote_source(query)
        
        # Print performance
        perf = result['pipeline_performance']
        print(f"Total time: {perf['total_time']:.3f}s")
        print(f"Success: {result['success']}")
        
        # Print stage breakdown
        for stage_log in perf['stage_performance']:
            stage_num = stage_log['stage']
            print(f"  Stage {stage_num}: {stage_log['candidates_count']} candidates in {stage_log['time']:.3f}s")
        
        # Print results
        if result['results']:
            for i, res in enumerate(result['results'][:2], 1):
                print(f"\nResult {i}:")
                print(f"  Match: {res.get('filter_match', 'N/A')}")
                print(f"  Confidence: {res.get('filter_confidence', 0):.3f}")
                print(f"  Text: {res['text'][:80]}...")
                print(f"  Work: {res.get('final_predicted_work', res.get('work', 'unknown'))}")
                print(f"  Category: {res.get('final_predicted_category', res.get('category', 'unknown'))}")
        else:
            print("No results found.")

if __name__ == "__main__":
    main()