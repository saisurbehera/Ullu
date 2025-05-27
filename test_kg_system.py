#!/usr/bin/env python3
"""
Test script for Knowledge Graph system.
"""

import sys
sys.path.append('/home/sai/Desktop/ullu/src')

from knowledge_graph import SanskritEntityExtractor, SanskritKnowledgeBase, KGEnhancer

def test_entity_extraction():
    """Test entity extraction functionality."""
    print("=== Testing Entity Extraction ===\n")
    
    extractor = SanskritEntityExtractor()
    
    # Test sentences
    test_texts = [
        "namaḥ śivāya maheśvarāya",
        "rāmo rājavaraḥ sadā vijayate rāmo rājeśvaraḥ",
        "kṛṣṇāya vāsudevāya haraye paramātmane",
        "dharmaṃ rakṣati rakṣitaḥ",
        "kalidāsasya meghadūte yakṣaḥ",
        "mahābhārate bhīṣmaparvani śrībhagavadgītā"
    ]
    
    for text in test_texts:
        print(f"Text: {text}")
        entities = extractor.extract_entities(text)
        
        for entity_type, entity_list in entities.items():
            if entity_list:
                print(f"  {entity_type.capitalize()}:")
                for matched, canonical, start, end in entity_list:
                    print(f"    - '{matched}' → '{canonical}' (pos: {start}-{end})")
        
        # Get summary
        summary = extractor.get_entity_summary(text)
        print(f"  Summary: {summary}")
        print()

def test_knowledge_base():
    """Test knowledge base functionality."""
    print("\n=== Testing Knowledge Base ===\n")
    
    kb = SanskritKnowledgeBase()
    
    # Test entity relations
    print("Relations for 'kṛṣṇa':")
    relations = kb.get_related_entities('kṛṣṇa')
    for relation, entity in relations[:5]:
        print(f"  - {relation}: {entity}")
    
    print("\nWorks containing 'kṛṣṇa':")
    works = kb.get_works_by_entity('kṛṣṇa')
    for work in works:
        work_info = kb.get_work_info(work)
        if work_info:
            print(f"  - {work_info['title']} by {work_info.get('author', 'Unknown')}")
    
    print("\nEntities in 'bhagavadgītā':")
    entities = kb.get_entities_in_work('bhagavadgītā')
    print(f"  {entities}")
    
    print("\nPath between 'arjuna' and 'vyāsa':")
    path = kb.get_path_between_entities('arjuna', 'vyāsa')
    if path:
        for e1, rel, e2 in path:
            print(f"  {e1} --[{rel}]--> {e2}")
    else:
        print("  No path found")

def test_kg_enhancer():
    """Test KG enhancer functionality."""
    print("\n=== Testing KG Enhancer ===\n")
    
    enhancer = KGEnhancer()
    
    # Test query expansion
    queries = [
        "viṣṇu stuti",
        "dharma rakṣati",
        "arjuna yuddha"
    ]
    
    for query in queries:
        print(f"Original query: '{query}'")
        expanded = enhancer.expand_query(query)
        print(f"Expanded queries:")
        for exp in expanded:
            print(f"  - {exp}")
        print()
    
    # Test KG features
    print("Testing KG features for query-passage pair:")
    query = "kṛṣṇa arjuna saṃvāda"
    passage = "bhagavadgītā mahābhārate kṛṣṇārjunasaṃvāde yogaśāstra"
    
    features = enhancer.compute_kg_features(query, passage)
    print(f"Query: '{query}'")
    print(f"Passage: '{passage}'")
    print("KG Features:")
    for feat, value in sorted(features.items()):
        if value > 0:
            print(f"  {feat}: {value:.3f}")
    
    # Test entity context
    print("\nEntity context for 'kṛṣṇa':")
    context = enhancer.get_entity_context('kṛṣṇa')
    print(f"  Works: {context['works']}")
    print(f"  Relations: {len(context['relations'])} found")
    if context['work_details']:
        print(f"  First work: {context['work_details'][0]['title']}")

def test_integration_example():
    """Test integrated example with retrieval enhancement."""
    print("\n=== Testing Integration Example ===\n")
    
    enhancer = KGEnhancer()
    
    # Simulate retrieval candidates
    candidates = [
        {
            'text': 'dharmo rakṣati rakṣitaḥ',
            'score': 0.8,
            'metadata': {'work': 'Manusmṛti', 'chapter': '8'}
        },
        {
            'text': 'kṛṣṇa uvāca pārtha paśyaitān samavetān kurūn',
            'score': 0.75,
            'metadata': {'work': 'Bhagavadgītā', 'chapter': '1'}
        },
        {
            'text': 'rāmo rājavaraḥ sadā vijayate',
            'score': 0.7,
            'metadata': {'work': 'Rāmacaritamānasa', 'chapter': '1'}
        }
    ]
    
    query = "kṛṣṇa arjuna"
    
    print(f"Query: '{query}'")
    print("\nOriginal candidates:")
    for i, cand in enumerate(candidates):
        print(f"  {i+1}. Score: {cand['score']:.2f} - {cand['text'][:50]}...")
    
    # Enhance candidates
    enhanced = enhancer.enhance_candidates(query, candidates)
    
    print("\nEnhanced candidates (with KG scores):")
    for i, cand in enumerate(enhanced):
        print(f"  {i+1}. Combined: {cand['combined_score']:.3f} "
              f"(Original: {cand['score']:.2f}, KG: {cand['kg_score']:.3f})")
        print(f"     Text: {cand['text'][:50]}...")
        print(f"     Work: {cand['metadata']['work']}")

if __name__ == "__main__":
    test_entity_extraction()
    test_knowledge_base()
    test_kg_enhancer()
    test_integration_example()
    
    print("\n✅ All KG tests completed!")