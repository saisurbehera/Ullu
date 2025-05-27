#!/usr/bin/env python3
"""Test script to verify index caching functionality."""

import time
import subprocess
import os
from pathlib import Path

def test_index_caching():
    """Test that index caching works correctly."""
    
    cache_path = Path("data/cached_multi_index.pkl")
    hash_path = Path("data/corpus_hash.json")
    
    # Remove existing cache files if any
    if cache_path.exists():
        os.remove(cache_path)
        print("Removed existing cache file")
    
    if hash_path.exists():
        os.remove(hash_path)
        print("Removed existing hash file")
    
    print("\n=== First Run (Building Index) ===")
    start_time = time.time()
    
    # Start the Flask app and let it initialize
    process = subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Read output until we see the server started or timeout
    timeout = 300  # 5 minutes timeout
    start = time.time()
    
    while time.time() - start < timeout:
        line = process.stdout.readline()
        if line:
            print(line.strip())
            
            # Check if index was built and saved
            if "Search indices cached successfully" in line:
                print(f"\n✅ Index built and cached successfully!")
            
            # Check if server started
            if "Running on" in line or "Serving Flask app" in line:
                break
    
    # Kill the process
    process.terminate()
    process.wait()
    
    first_run_time = time.time() - start_time
    print(f"\nFirst run time: {first_run_time:.2f} seconds")
    
    # Check if cache files were created
    if cache_path.exists() and hash_path.exists():
        print(f"✅ Cache files created successfully!")
        print(f"   - Index size: {cache_path.stat().st_size / 1024 / 1024:.2f} MB")
    else:
        print("❌ Cache files not created!")
        return
    
    print("\n=== Second Run (Loading from Cache) ===")
    start_time = time.time()
    
    # Start the Flask app again
    process = subprocess.Popen(
        ["python", "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True
    )
    
    # Read output
    start = time.time()
    cache_loaded = False
    
    while time.time() - start < timeout:
        line = process.stdout.readline()
        if line:
            print(line.strip())
            
            # Check if index was loaded from cache
            if "Cached search indices loaded successfully" in line:
                print(f"\n✅ Index loaded from cache successfully!")
                cache_loaded = True
            
            # Check if server started
            if "Running on" in line or "Serving Flask app" in line:
                break
    
    # Kill the process
    process.terminate()
    process.wait()
    
    second_run_time = time.time() - start_time
    print(f"\nSecond run time: {second_run_time:.2f} seconds")
    
    if cache_loaded:
        speedup = first_run_time / second_run_time
        print(f"\n🚀 Speedup: {speedup:.2f}x faster with caching!")
    else:
        print("\n❌ Cache was not loaded on second run")

if __name__ == "__main__":
    test_index_caching()