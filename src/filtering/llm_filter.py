import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import json
import re
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class FilterResult:
    """Result from LLM filtering stage."""
    match: bool
    confidence: float
    predicted_work: Optional[str] = None
    predicted_category: Optional[str] = None
    predicted_verse: Optional[str] = None
    reasoning: Optional[str] = None

class RuleBasedFilter:
    """
    Rule-based filtering system as fallback for LLM.
    Implements heuristics for Sanskrit quote matching.
    """
    
    def __init__(self):
        self.work_patterns = {
            'bhagavadgītā': [r'gītā', r'bhagavad', r'kṛṣṇa.*arjuna', r'kurukṣetra'],
            'rāmāyaṇa': [r'rāma', r'sītā', r'hanumān', r'rāvaṇa', r'laṅkā'],
            'mahābhārata': [r'bhārata', r'pāṇḍava', r'kaurava', r'yudhiṣṭhira'],
            'purāṇa': [r'purāṇa', r'brahmā', r'viṣṇu', r'śiva'],
            'upaniṣad': [r'upaniṣad', r'brahman', r'ātman', r'om'],
            'vedas': [r'veda', r'mantra', r'yajña', r'soma']
        }
        
        self.genre_indicators = {
            'epic': [r'yuddha', r'vīra', r'rāja', r'śastra'],
            'devotional': [r'bhakti', r'prema', r'sevā', r'guru'],
            'philosophical': [r'tattva', r'jñāna', r'darśana', r'mokṣa'],
            'ritual': [r'yajña', r'pūjā', r'mantra', r'homa']
        }
    
    def extract_metadata(self, text: str) -> Dict[str, str]:
        """Extract metadata from source citations in text."""
        metadata = {}
        
        # Look for common citation patterns
        citations = re.findall(r'//\s*([^/]+)\s*//', text)
        if citations:
            citation = citations[0].strip()
            metadata['citation'] = citation
            
            # Extract work abbreviations
            work_match = re.search(r'([A-Za-z]+P?)_?(\d+)?', citation)
            if work_match:
                metadata['work_abbrev'] = work_match.group(1)
        
        # Look for verse numbers
        verse_pattern = r'(\d+)\.(\d+)'
        verse_match = re.search(verse_pattern, text)
        if verse_match:
            metadata['chapter'] = verse_match.group(1)
            metadata['verse'] = verse_match.group(2)
        
        return metadata
    
    def identify_work(self, query: str, passage: str) -> Optional[str]:
        """Identify the most likely work based on content."""
        combined_text = (query + ' ' + passage).lower()
        
        work_scores = {}
        
        for work, patterns in self.work_patterns.items():
            score = 0
            for pattern in patterns:
                matches = len(re.findall(pattern, combined_text))
                score += matches
            
            if score > 0:
                work_scores[work] = score
        
        if work_scores:
            return max(work_scores, key=work_scores.get)
        
        return None
    
    def calculate_match_confidence(self, query: str, passage: str) -> float:
        """Calculate confidence that passage matches query."""
        confidence = 0.0
        
        # Exact substring match
        if query.lower() in passage.lower():
            confidence += 0.5
        
        # Word overlap
        query_words = set(query.lower().split())
        passage_words = set(passage.lower().split())
        
        if query_words:
            overlap = len(query_words.intersection(passage_words))
            word_confidence = overlap / len(query_words)
            confidence += word_confidence * 0.3
        
        # Length similarity
        length_ratio = min(len(query), len(passage)) / max(len(query), len(passage))
        if length_ratio > 0.3:
            confidence += 0.2
        
        return min(confidence, 1.0)
    
    def filter(self, query: str, passage: str, passage_metadata: Dict) -> FilterResult:
        """Apply rule-based filtering."""
        # Extract metadata from text
        text_metadata = self.extract_metadata(passage)
        
        # Calculate match confidence
        confidence = self.calculate_match_confidence(query, passage)
        
        # Determine if it's a match
        is_match = confidence > 0.5
        
        # Identify work
        predicted_work = self.identify_work(query, passage)
        if not predicted_work and 'work' in passage_metadata:
            predicted_work = passage_metadata['work']
        
        # Extract category
        predicted_category = passage_metadata.get('category', 'unknown')
        
        # Combine verse information
        predicted_verse = None
        if 'citation' in text_metadata:
            predicted_verse = text_metadata['citation']
        elif 'chapter' in text_metadata and 'verse' in text_metadata:
            predicted_verse = f"{text_metadata['chapter']}.{text_metadata['verse']}"
        
        return FilterResult(
            match=is_match,
            confidence=confidence,
            predicted_work=predicted_work,
            predicted_category=predicted_category,
            predicted_verse=predicted_verse,
            reasoning=f"Rule-based filter: confidence={confidence:.3f}"
        )

class MockLLMFilter:
    """
    Mock LLM filter that simulates GPT-style reasoning.
    In production, this would use actual LLM API calls.
    """
    
    def __init__(self):
        self.rule_filter = RuleBasedFilter()
    
    def create_prompt(self, query: str, passage: str, passage_metadata: Dict) -> str:
        """Create prompt for LLM filtering."""
        prompt = f"""
Given a Sanskrit quote query and a candidate passage, determine if they match and extract bibliographic information.

QUERY: "{query}"

CANDIDATE PASSAGE: "{passage}"

PASSAGE METADATA:
- Source file: {passage_metadata.get('source_file', 'unknown')}
- Work: {passage_metadata.get('work', 'unknown')}  
- Category: {passage_metadata.get('category', 'unknown')}

TASK: Analyze if the passage is the correct source for the query. Provide:

1. MATCH: "yes" or "no" - whether this passage contains or is the source of the query
2. CONFIDENCE: 0.0-1.0 - confidence in the match
3. WORK: Identified Sanskrit work name
4. CATEGORY: Text category (epic, purāṇa, upaniṣad, etc.)
5. VERSE_REFERENCE: Specific verse/chapter reference if available
6. REASONING: Brief explanation of the decision

Respond in JSON format:
{{
    "match": "yes/no",
    "confidence": 0.0-1.0,
    "predicted_work": "work name",
    "predicted_category": "category",
    "predicted_verse": "verse reference",
    "reasoning": "explanation"
}}
"""
        return prompt
    
    def simulate_llm_response(self, query: str, passage: str, passage_metadata: Dict) -> Dict:
        """Simulate LLM response using rule-based logic."""
        # Use rule-based filter as base
        rule_result = self.rule_filter.filter(query, passage, passage_metadata)
        
        # Simulate some LLM-style reasoning
        reasoning_parts = []
        
        if query.lower() in passage.lower():
            reasoning_parts.append("Query appears as substring in passage")
        
        query_words = set(query.lower().split())
        passage_words = set(passage.lower().split())
        overlap = len(query_words.intersection(passage_words))
        
        if overlap > 0:
            reasoning_parts.append(f"Word overlap: {overlap}/{len(query_words)} words")
        
        if rule_result.predicted_work:
            reasoning_parts.append(f"Identified work: {rule_result.predicted_work}")
        
        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No clear indicators"
        
        response = {
            "match": "yes" if rule_result.match else "no",
            "confidence": rule_result.confidence,
            "predicted_work": rule_result.predicted_work or "unknown",
            "predicted_category": rule_result.predicted_category or "unknown", 
            "predicted_verse": rule_result.predicted_verse,
            "reasoning": reasoning
        }
        
        return response
    
    def filter(self, query: str, passage: str, passage_metadata: Dict) -> FilterResult:
        """Apply LLM-based filtering."""
        try:
            # In production, this would make an API call to LLM
            response = self.simulate_llm_response(query, passage, passage_metadata)
            
            return FilterResult(
                match=response["match"] == "yes",
                confidence=float(response["confidence"]),
                predicted_work=response.get("predicted_work"),
                predicted_category=response.get("predicted_category"),
                predicted_verse=response.get("predicted_verse"),
                reasoning=response.get("reasoning")
            )
            
        except Exception as e:
            logger.error(f"Error in LLM filtering: {e}")
            # Fallback to rule-based
            return self.rule_filter.filter(query, passage, passage_metadata)

class Stage3Filter:
    """
    Stage 3: LLM-based filtering and final confirmation.
    Goal: Binary accept/reject and granular label prediction.
    """
    
    def __init__(self, use_llm: bool = True):
        self.use_llm = use_llm
        
        if use_llm:
            self.filter = MockLLMFilter()
        else:
            self.filter = RuleBasedFilter()
    
    def filter_candidates(self, query: str, candidates: List[Dict], 
                         top_k: int = 1) -> List[Dict]:
        """Filter and rank candidates, returning top-k results."""
        filtered_results = []
        
        for candidate in candidates:
            passage = candidate.get('text', '')
            metadata = {
                'source_file': candidate.get('source_file', ''),
                'work': candidate.get('work', ''),
                'category': candidate.get('category', ''),
                'quote_id': candidate.get('quote_id', '')
            }
            
            # Apply filtering
            filter_result = self.filter.filter(query, passage, metadata)
            
            # Add filter results to candidate
            enhanced_candidate = candidate.copy()
            enhanced_candidate.update({
                'filter_match': filter_result.match,
                'filter_confidence': filter_result.confidence,
                'final_predicted_work': filter_result.predicted_work,
                'final_predicted_category': filter_result.predicted_category,
                'final_predicted_verse': filter_result.predicted_verse,
                'filter_reasoning': filter_result.reasoning
            })
            
            filtered_results.append(enhanced_candidate)
        
        # Sort by filter confidence
        filtered_results.sort(key=lambda x: x['filter_confidence'], reverse=True)
        
        # Filter to only matches and return top-k
        matches = [r for r in filtered_results if r['filter_match']]
        
        return matches[:top_k] if matches else filtered_results[:top_k]
    
    def evaluate_filtering(self, test_cases: List[Dict]) -> Dict[str, float]:
        """Evaluate filtering performance."""
        correct_matches = 0
        total_cases = len(test_cases)
        
        for case in test_cases:
            query = case['query']
            passage = case['passage']
            metadata = case.get('metadata', {})
            expected_match = case.get('expected_match', True)
            
            result = self.filter.filter(query, passage, metadata)
            
            if result.match == expected_match:
                correct_matches += 1
        
        accuracy = correct_matches / total_cases if total_cases > 0 else 0.0
        
        return {
            'accuracy': accuracy,
            'correct': correct_matches,
            'total': total_cases
        }

def main():
    """Test Stage 3 filtering system."""
    logging.basicConfig(level=logging.INFO)
    
    # Initialize filter
    stage3_filter = Stage3Filter(use_llm=True)
    
    # Test cases
    test_candidates = [
        {
            'text': 'dharmaviṣṇupriya śrīkṛṣṇa paramātma sanātana // BG_4.7 //',
            'work': 'bhagavadgītā',
            'category': 'epic',
            'quote_id': 1,
            'ranking_score': 0.95
        },
        {
            'text': 'rājya dhana āryāṇām viṣṇave namaḥ',
            'work': 'unknown',
            'category': 'purāṇa',
            'quote_id': 2,
            'ranking_score': 0.85
        },
        {
            'text': 'gaṅgā yamunā sarasvatī devī pārvatī lakṣmī',
            'work': 'stotras',
            'category': 'devotional',
            'quote_id': 3,
            'ranking_score': 0.75
        }
    ]
    
    test_query = "dharmaviṣṇupriya śrīkṛṣṇa"
    
    print("\n" + "="*60)
    print("STAGE 3 FILTERING TEST")
    print("="*60)
    print(f"Query: '{test_query}'")
    print("-" * 40)
    
    # Filter candidates
    filtered_results = stage3_filter.filter_candidates(test_query, test_candidates, top_k=2)
    
    for i, result in enumerate(filtered_results, 1):
        print(f"\nResult {i}:")
        print(f"  Match: {result['filter_match']}")
        print(f"  Confidence: {result['filter_confidence']:.3f}")
        print(f"  Text: {result['text'][:60]}...")
        print(f"  Predicted Work: {result['final_predicted_work']}")
        print(f"  Predicted Category: {result['final_predicted_category']}")
        print(f"  Reasoning: {result['filter_reasoning']}")
    
    # Test evaluation
    eval_cases = [
        {
            'query': 'dharma viṣṇu',
            'passage': 'dharmaviṣṇupriya śrīkṛṣṇa paramātma',
            'metadata': {'work': 'bhagavadgītā'},
            'expected_match': True
        },
        {
            'query': 'rāma sītā',
            'passage': 'kṛṣṇa arjuna battlefield',
            'metadata': {'work': 'bhagavadgītā'},
            'expected_match': False
        }
    ]
    
    eval_results = stage3_filter.evaluate_filtering(eval_cases)
    
    print(f"\n" + "="*60)
    print("FILTERING EVALUATION")
    print("="*60)
    print(f"Accuracy: {eval_results['accuracy']:.1%}")
    print(f"Correct: {eval_results['correct']}/{eval_results['total']}")

if __name__ == "__main__":
    main()