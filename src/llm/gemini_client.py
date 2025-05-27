#!/usr/bin/env python3
"""
Gemini API client for Sanskrit quote retrieval pipeline.
Handles LLM-based filtering and confirmation tasks.
"""

import os
import logging
from typing import List, Dict, Optional
from pathlib import Path
from dotenv import load_dotenv

try:
    from google import genai
except ImportError:
    print("Google GenAI SDK not installed. Run: pip install google-genai")
    genai = None

logger = logging.getLogger(__name__)

class GeminiClient:
    """Client for Google Gemini API integration."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Gemini client with API key."""
        # Load environment variables
        load_dotenv()
        
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('GEMINI_API_KEY')
        
        if not self.api_key or self.api_key == 'your_api_key_here':
            raise ValueError(
                "Gemini API key not found. Please set GEMINI_API_KEY in .env file or pass api_key parameter."
            )
        
        if genai is None:
            raise ImportError("Google GenAI SDK not available. Install with: pip install google-genai")
        
        # Initialize client
        self.client = genai.Client(api_key=self.api_key)
        self.model = "gemini-2.0-flash"
        
        logger.info("Gemini client initialized successfully")
    
    def generate_content(self, prompt: str, **kwargs) -> str:
        """Generate content using Gemini API."""
        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt,
                **kwargs
            )
            return response.text
        except Exception as e:
            logger.error(f"Gemini API error: {e}")
            raise
    
    def filter_relevant_passages(self, query: str, passages: List[str], 
                                context: str = "Sanskrit quotes") -> List[Dict]:
        """Use Gemini to filter and score passage relevance."""
        
        prompt = f"""You are an expert in {context} and semantic matching. 

Query: "{query}"

Please evaluate the following passages for relevance to the query. For each passage, provide:
1. Relevance score (0-10)
2. Brief explanation
3. Key matching concepts

Passages:
"""
        
        for i, passage in enumerate(passages, 1):
            prompt += f"\n{i}. {passage}\n"
        
        prompt += """
Please respond in this JSON format:
{
  "evaluations": [
    {
      "passage_number": 1,
      "relevance_score": 8,
      "explanation": "Strong thematic connection...",
      "key_concepts": ["dharma", "duty"]
    }
  ]
}
"""
        
        try:
            response = self.generate_content(prompt)
            # Parse JSON response (add error handling as needed)
            import json
            result = json.loads(response)
            return result.get('evaluations', [])
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {e}")
            # Fallback: return basic scores with better explanation
            return [{"passage_number": i+1, "relevance_score": 6, 
                    "explanation": f"LLM response parsing failed - using default score"} 
                   for i in range(len(passages))]
        except Exception as e:
            logger.error(f"Error in passage filtering: {e}")
            # Fallback: return basic scores
            return [{"passage_number": i+1, "relevance_score": 5, 
                    "explanation": f"LLM evaluation failed: {str(e)}"} 
                   for i in range(len(passages))]
    
    def confirm_semantic_match(self, query: str, passage: str, 
                              threshold: float = 6.0) -> Dict:
        """Use Gemini to confirm if passage semantically matches query."""
        
        prompt = f"""As a Sanskrit and semantic analysis expert, determine if this passage is relevant to the query.

Query: "{query}"
Passage: "{passage}"

Consider:
1. Direct word matches
2. Semantic/thematic connections
3. Conceptual relationships
4. Cultural/religious context

Provide a relevance score (0-10) and explanation.

Respond in JSON format:
{{
  "relevance_score": 7.5,
  "is_relevant": true,
  "explanation": "The passage discusses similar themes...",
  "matching_concepts": ["dharma", "righteousness"],
  "confidence": "high"
}}
"""
        
        try:
            response = self.generate_content(prompt)
            import json
            result = json.loads(response)
            result['is_relevant'] = result.get('relevance_score', 0) >= threshold
            return result
        except Exception as e:
            logger.error(f"Error in semantic matching: {e}")
            return {
                "relevance_score": 5.0,
                "is_relevant": False,
                "explanation": f"Error in LLM evaluation: {e}",
                "confidence": "low"
            }
    
    def generate_query_variations(self, original_query: str, num_variations: int = 5) -> List[str]:
        """Generate query variations for better retrieval coverage."""
        
        prompt = f"""Generate {num_variations} variations of this Sanskrit-related query to improve search coverage:

Original query: "{original_query}"

Create variations that:
1. Use synonyms (Sanskrit and English)
2. Rephrase concepts
3. Add context
4. Use different transliterations
5. Include related themes

Return only the variations, one per line.
"""
        
        try:
            response = self.generate_content(prompt)
            variations = [line.strip() for line in response.strip().split('\n') 
                         if line.strip() and line.strip() != original_query]
            return variations[:num_variations]
        except Exception as e:
            logger.error(f"Error generating query variations: {e}")
            return []


def test_gemini_client():
    """Test Gemini client functionality."""
    try:
        client = GeminiClient()
        
        # Test basic generation
        response = client.generate_content("Explain dharma in one sentence.")
        print(f"Basic test: {response}")
        
        # Test semantic matching
        query = "righteousness and duty"
        passage = "dharmaṃ cara tapaś cara"
        match_result = client.confirm_semantic_match(query, passage)
        print(f"Semantic match: {match_result}")
        
        # Test query variations
        variations = client.generate_query_variations("dharma")
        print(f"Query variations: {variations}")
        
        print("✅ Gemini client test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Gemini client test failed: {e}")
        return False


if __name__ == "__main__":
    test_gemini_client()