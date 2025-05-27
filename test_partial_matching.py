#!/usr/bin/env python3
"""
Test partial text matching functionality.
"""

import sys
sys.path.append('/home/sai/Desktop/ullu/src')

from indexing.partial_match_index import PartialMatchIndex
from indexing.multi_index import MultiIndex
import pandas as pd

def test_partial_matching():
    """Test partial text matching on Sanskrit quotes."""
    print("=== Testing Partial Text Matching ===\n")
    
    # Test data with full verses
    test_documents = [
        "satyaṃ brūhi sutaḥ kasya somasyātha bṛhaspateḥ",
        "dharmo rakṣati rakṣitaḥ yaḥ dharmaṃ rakṣati sa eva rakṣyate",
        "vidyā dadāti vinayaṃ vinayād yāti pātratām",
        "kāmadhenuguṇā vidyā hy akāle phalam ucyate",
        "na hi jñānena sadṛśaṃ pavitram iha vidyate"
    ]
    
    # Create partial match index
    partial_index = PartialMatchIndex(window_size=10, overlap=5)
    partial_index.fit(test_documents)
    
    # Test queries
    test_queries = [
        ("somasyātha bṛhaspateḥ", "Partial from first verse"),
        ("dharmo rakṣati", "Beginning of second verse"),
        ("rakṣati rakṣitaḥ", "Middle of second verse"),
        ("phalam ucyate", "End of fourth verse"),
        ("jñānena sadṛśaṃ", "Middle of fifth verse")
    ]
    
    print("Test Documents:")
    for i, doc in enumerate(test_documents):
        print(f"{i}: {doc}")
    
    print("\n" + "="*50 + "\n")
    
    for query, description in test_queries:
        print(f"Query: '{query}' ({description})")
        results = partial_index.search(query, top_k=3)
        
        if results:
            for doc_id, score in results[:3]:
                print(f"  Doc {doc_id}: Score {score:.3f}")
                print(f"  Text: {test_documents[doc_id]}")
        else:
            print("  No matches found")
        print()

def test_with_real_data():
    """Test partial matching with real Sanskrit corpus."""
    print("\n=== Testing with Real Sanskrit Data ===\n")
    
    # Load a sample of the corpus
    try:
        df = pd.read_csv('/home/sai/Desktop/ullu/data/sanskrit_quotes.csv')
        sample_size = min(100, len(df))
        sample_df = df.head(sample_size)
        documents = sample_df['text'].tolist()
        
        print(f"Loaded {len(documents)} documents for testing")
        
        # Create multi-index with partial matching
        multi_index = MultiIndex(config={'use_partial': True, 'window_size': 15, 'overlap': 7})
        multi_index.fit(documents)
        
        # Test the specific case mentioned by user
        query = "somasyātha bṛhaspateḥ"
        print(f"\nSearching for: '{query}'")
        
        # Search with partial matching enabled
        results_with_partial = multi_index.search(query, top_k=5, use_partial=True)
        print("\nResults WITH partial matching:")
        for i, result in enumerate(results_with_partial[:5]):
            print(f"\n{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            if result.get('partial_match'):
                print(f"   Partial Match: Yes (score: {result.get('partial_score', 0):.3f})")
        
        # Search without partial matching
        results_without_partial = multi_index.search(query, top_k=5, use_partial=False)
        print("\n\nResults WITHOUT partial matching:")
        for i, result in enumerate(results_without_partial[:5]):
            print(f"\n{i+1}. Score: {result['score']:.3f}")
            print(f"   Text: {result['text'][:100]}...")
            
    except Exception as e:
        print(f"Error loading data: {e}")

def test_edge_cases():
    """Test edge cases for partial matching."""
    print("\n=== Testing Edge Cases ===\n")
    
    documents = [
        "short text",
        "a" * 50,  # Repetitive text
        "वाचं वर्धयन्ति तं विधिं वेदयन्ति",  # Devanagari
        "mixed वाचं english विधिं text",
        ""  # Empty document
    ]
    
    partial_index = PartialMatchIndex(window_size=5, overlap=2)
    partial_index.fit(documents)
    
    queries = [
        "short",
        "aaaa",
        "वाचं",
        "english विधिं",
        "nonexistent"
    ]
    
    for query in queries:
        print(f"Query: '{query}'")
        results = partial_index.search(query, top_k=2)
        if results:
            for doc_id, score in results:
                print(f"  Doc {doc_id}: Score {score:.3f} - {documents[doc_id][:50]}")
        else:
            print("  No matches")
        print()

if __name__ == "__main__":
    test_partial_matching()
    test_with_real_data()
    test_edge_cases()
    
    print("\n✅ Partial matching tests completed!")