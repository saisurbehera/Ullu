#!/usr/bin/env python3
"""Test Gemini integration with search."""

import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import logging
logging.basicConfig(level=logging.DEBUG)

from llm.gemini_client import GeminiClient

def test_gemini_search():
    """Test Gemini for search result reranking."""
    print("=== Testing Gemini Search Enhancement ===\n")
    
    # Initialize Gemini client
    try:
        gemini_client = GeminiClient()
        print("✅ Gemini client initialized successfully")
    except Exception as e:
        print(f"❌ Failed to initialize Gemini: {e}")
        return
    
    # Test passages
    query = "dharma rakṣati"
    passages = [
        "dharmo rakṣati rakṣitaḥ",
        "vidyā dadāti vinayaṃ",
        "satyaṃ brūhi priyaṃ brūhi",
        "dharmaṃ carati yaḥ sadā"
    ]
    
    print(f"\nQuery: '{query}'")
    print("\nPassages to rank:")
    for i, p in enumerate(passages):
        print(f"  {i+1}. {p}")
    
    # Get Gemini rankings
    try:
        print("\n🤖 Getting Gemini relevance scores...")
        llm_scores = gemini_client.filter_relevant_passages(query, passages)
        
        print("\nGemini Results:")
        for i, score_info in enumerate(llm_scores):
            print(f"\n{i+1}. Passage: {passages[i]}")
            print(f"   Relevance Score: {score_info.get('relevance_score', 'N/A')}/10")
            print(f"   Explanation: {score_info.get('explanation', 'N/A')}")
            
    except Exception as e:
        print(f"\n❌ Gemini scoring failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gemini_search()