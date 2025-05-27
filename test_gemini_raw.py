#!/usr/bin/env python3
"""Test raw Gemini response."""

import sys
sys.path.append('/home/sai/Desktop/ullu/src')

from llm.gemini_client import GeminiClient

def test_raw_response():
    """Test what Gemini actually returns."""
    
    gemini = GeminiClient()
    
    # Simple test prompt
    prompt = "Return a JSON object with a single field 'test' set to 'success'."
    
    print(f"Prompt: {prompt}")
    print("\nRaw response:")
    print("-" * 50)
    
    response = gemini.generate_content(prompt)
    print(response)
    print("-" * 50)
    print(f"\nResponse type: {type(response)}")
    print(f"Response length: {len(response)} characters")

if __name__ == "__main__":
    test_raw_response()