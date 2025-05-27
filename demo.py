#!/usr/bin/env python3
"""
Simple demo of the Sanskrit Quote Retrieval Pipeline (ULLU)
Demonstrates the 3-stage approach without complex dependencies.
"""

import sys
import pandas as pd
import time
from pathlib import Path

# Add src to path
sys.path.append('src')

def demo_stage1_only():
    """Demo Stage 1 retrieval only."""
    print("🔍 ULLU - Sanskrit Quote Retrieval Demo")
    print("=" * 50)
    
    # Simple BM25-like scoring
    def simple_score(query, text):
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if not query_words:
            return 0.0
        
        overlap = len(query_words.intersection(text_words))
        return overlap / len(query_words)
    
    # Load data
    print("📚 Loading Sanskrit corpus...")
    df = pd.read_csv('data/sanskrit_quotes.csv')
    print(f"   Loaded {len(df)} Sanskrit passages")
    
    # Test queries
    test_queries = [
        "dharma",
        "viṣṇu hari", 
        "devī cakre bhayākulā",
        "namaḥ svadhāyai",
        "rājasya"
    ]
    
    print("\n🔍 Testing Quote Retrieval:")
    print("-" * 50)
    
    for query in test_queries:
        print(f"\n📖 Query: '{query}'")
        
        start_time = time.time()
        
        # Score all passages
        scores = []
        for idx, row in df.iterrows():
            text = str(row['text'])
            score = simple_score(query, text)
            if score > 0:
                scores.append({
                    'score': score,
                    'text': text,
                    'work': row.get('work', 'unknown'),
                    'category': row.get('category', 'unknown')
                })
        
        # Sort by score
        scores.sort(key=lambda x: x['score'], reverse=True)
        
        retrieval_time = time.time() - start_time
        
        print(f"   ⏱️  Retrieved {len(scores)} candidates in {retrieval_time:.3f}s")
        
        # Show top 3 results
        for i, result in enumerate(scores[:3], 1):
            print(f"   {i}. Score: {result['score']:.3f}")
            print(f"      Text: {result['text'][:70]}...")
            print(f"      Source: {result['work']} ({result['category']})")
    
    # Show corpus statistics
    print(f"\n📊 Corpus Statistics:")
    print(f"   Total passages: {len(df)}")
    print(f"   Categories: {', '.join(df['category'].unique())}")
    print(f"   Average length: {df['length'].mean():.0f} characters")
    print(f"   Top works: {', '.join(df['work'].value_counts().head(3).index.tolist())}")

def demo_simple_pipeline():
    """Demo with simple 3-stage pipeline."""
    print("\n🚀 3-Stage Pipeline Demo")
    print("=" * 50)
    
    # Load data
    df = pd.read_csv('data/sanskrit_quotes.csv')
    
    query = "devī cakre bhayākulā"
    print(f"📖 Query: '{query}'")
    
    # Stage 1: Simple retrieval
    print("\n🎯 Stage 1: Retrieval")
    candidates = []
    
    for idx, row in df.iterrows():
        text = str(row['text'])
        
        # Simple word overlap score
        query_words = set(query.lower().split())
        text_words = set(text.lower().split())
        
        if query_words:
            overlap = len(query_words.intersection(text_words))
            score = overlap / len(query_words)
            
            if score > 0:
                candidates.append({
                    'text': text,
                    'work': row.get('work', 'unknown'),
                    'category': row.get('category', 'unknown'),
                    'stage1_score': score,
                    'quote_id': row.get('quote_id', idx)
                })
    
    candidates.sort(key=lambda x: x['stage1_score'], reverse=True)
    stage1_top = candidates[:10]
    
    print(f"   Found {len(candidates)} candidates, keeping top 10")
    print(f"   Best match: {stage1_top[0]['text'][:50]}... (score: {stage1_top[0]['stage1_score']:.3f})")
    
    # Stage 2: Simple ranking (length + exact match bonus)
    print("\n🔀 Stage 2: Ranking")
    
    for candidate in stage1_top:
        text = candidate['text'].lower()
        
        # Exact substring bonus
        exact_bonus = 0.5 if query.lower() in text else 0.0
        
        # Length similarity bonus  
        query_len = len(query)
        text_len = len(candidate['text'])
        length_sim = min(query_len, text_len) / max(query_len, text_len)
        
        # Combined ranking score
        candidate['stage2_score'] = candidate['stage1_score'] + exact_bonus + (length_sim * 0.3)
    
    stage1_top.sort(key=lambda x: x['stage2_score'], reverse=True)
    stage2_top = stage1_top[:3]
    
    print(f"   Re-ranked top 3 candidates")
    print(f"   Best match: {stage2_top[0]['text'][:50]}... (score: {stage2_top[0]['stage2_score']:.3f})")
    
    # Stage 3: Simple filtering
    print("\n✅ Stage 3: Filtering")
    
    final_results = []
    for candidate in stage2_top:
        # Simple match decision
        is_match = candidate['stage2_score'] > 0.7
        
        candidate['final_match'] = is_match
        candidate['confidence'] = min(candidate['stage2_score'], 1.0)
        
        if is_match:
            final_results.append(candidate)
    
    print(f"   Filtered to {len(final_results)} final matches")
    
    # Show final results
    print(f"\n🎯 Final Results:")
    print("-" * 30)
    
    if final_results:
        for i, result in enumerate(final_results, 1):
            print(f"\n{i}. Match: {result['final_match']}")
            print(f"   Confidence: {result['confidence']:.3f}")
            print(f"   Text: {result['text']}")
            print(f"   Source: {result['work']} ({result['category']})")
    else:
        print("   No confident matches found.")
    
    print(f"\n✨ Pipeline completed successfully!")

def main():
    """Run the demo."""
    print("🕉️  ULLU - Sanskrit Quote Retrieval System")
    print("   3-Stage Pipeline for Identifying Sanskrit Quote Sources")
    print("   Stage 1: BM25 + N-gram Retrieval (Recall@100 ≥ 95%)")
    print("   Stage 2: Cross-encoder Ranking (Precision@1 maximized)")  
    print("   Stage 3: LLM-based Filtering & Source Identification")
    print()
    
    # Check if data exists
    if not Path('data/sanskrit_quotes.csv').exists():
        print("❌ Error: Sanskrit quotes dataset not found!")
        print("   Please run the data processing script first.")
        return
    
    # Run demos
    demo_stage1_only()
    demo_simple_pipeline()
    
    print(f"\n🙏 Demo completed. Thank you for trying ULLU!")
    print("   For the full pipeline with trained models, see src/pipeline.py")

if __name__ == "__main__":
    main()