#!/usr/bin/env python3
"""Test if Gemini loads in app context."""

import sys
sys.path.append('/home/sai/Desktop/ullu/src')

# Test the imports that app.py uses
print("Testing imports...")

try:
    from llm.gemini_client import GeminiClient
    print("✅ GeminiClient import successful")
    gemini_available = True
except ImportError as e:
    print(f"❌ GeminiClient import error: {e}")
    gemini_available = False

if gemini_available:
    try:
        gemini_client = GeminiClient()
        print("✅ Gemini client initialized successfully")
    except Exception as e:
        print(f"❌ Gemini client initialization failed: {e}")
        gemini_client = None
else:
    print("⚠️ Gemini client not available")
    gemini_client = None

print(f"\nFinal state: gemini_client = {gemini_client}")
print(f"Gemini available for use: {gemini_client is not None}")