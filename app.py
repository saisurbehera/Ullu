#!/usr/bin/env python3
"""
Simple web interface for Sanskrit Quote Retrieval (ULLU)
Flask backend for search functionality.
"""

import sys
sys.path.append('/home/sai/Desktop/ullu/src')

from flask import Flask, render_template, request, jsonify
import pandas as pd
import logging
from pathlib import Path
import traceback
from tqdm import tqdm
import json

# Import our retrieval components
try:
    from indexing.multi_index import MultiIndex
    multi_index_available = True
    search_type = "multi_index"
except ImportError as e:
    print(f"MultiIndex import error: {e}")
    try:
        from simple_search import SimpleSearch as MultiIndex
        multi_index_available = True
        search_type = "simple"
        print("Using simple search fallback")
    except ImportError as e2:
        print(f"Simple search import error: {e2}")
        multi_index_available = False
        search_type = "none"

try:
    from preprocessing.simple_normalizer import SimpleTextNormalizer as SanskritTextNormalizer
    normalizer_available = True
    normalizer_type = "simple"
    print("Using simple normalizer (no indic-transliteration)")
except ImportError as e:
    print(f"Simple normalizer import error: {e}")
    try:
        from preprocessing.text_normalizer import SanskritTextNormalizer
        normalizer_available = True
        normalizer_type = "full"
        print("Using full normalizer")
    except ImportError as e2:
        print(f"Full normalizer import error: {e2}")
        normalizer_available = False
        normalizer_type = "none"

try:
    from llm.gemini_client import GeminiClient
    gemini_available = True
except ImportError as e:
    print(f"GeminiClient import error: {e}")
    gemini_available = False

try:
    from utils.metadata_mapper import metadata_mapper
    metadata_available = True
except ImportError as e:
    print(f"Metadata mapper import error: {e}")
    metadata_available = False

app = Flask(__name__)
app.config['DEBUG'] = True

# Global variables for loaded components
multi_index = None
normalizer = None
gemini_client = None
quotes_df = None

# Index cache configuration
INDEX_CACHE_PATH = "data/cached_multi_index.pkl"
CORPUS_HASH_PATH = "data/corpus_hash.json"

def compute_corpus_hash(df):
    """Compute a hash of the corpus to detect changes."""
    import hashlib
    # Create a hash of corpus size and first/last few entries
    corpus_info = f"{len(df)}_{df['text'].iloc[0] if len(df) > 0 else ''}_{df['text'].iloc[-1] if len(df) > 0 else ''}"
    return hashlib.md5(corpus_info.encode()).hexdigest()

def load_components():
    """Load all retrieval components."""
    global multi_index, normalizer, gemini_client, quotes_df
    
    # Check GPU availability
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"🚀 GPU detected: {gpu_name} ({gpu_memory:.1f}GB)")
            print(f"   Available GPUs: {gpu_count}")
        else:
            print("💻 Running on CPU (no GPU detected)")
    except ImportError:
        print("💻 Running on CPU (PyTorch not available)")
    
    try:
        # Load full corpus dataset (prefer corpus over curated quotes)
        corpus_path = "data/sanskrit_corpus.csv"
        quotes_path = "data/sanskrit_quotes.csv"
        
        if Path(corpus_path).exists():
            print(f"📚 Loading FULL Sanskrit corpus (~2M entries)...")
            print("⚠️  This will take several minutes and use significant memory")
            
            # Use tqdm with pandas progress monitoring
            tqdm.pandas(desc="Loading corpus")
            quotes_df = pd.read_csv(corpus_path)
            print(f"✅ Loaded {len(quotes_df):,} passages from full corpus")
        elif Path(quotes_path).exists():
            quotes_df = pd.read_csv(quotes_path)
            print(f"✅ Loaded {len(quotes_df)} curated quotes")
        else:
            print("❌ No dataset found")
            return False
        
        # Initialize text normalizer
        if normalizer_available:
            normalizer = SanskritTextNormalizer()
            print(f"✅ Text normalizer initialized ({normalizer_type})")
        else:
            normalizer = None
            print("⚠️ Text normalizer not available - using basic text processing")
        
        # Initialize multi-index retrieval
        if multi_index_available:
            multi_index = MultiIndex()
            corpus_hash = compute_corpus_hash(quotes_df)
            index_loaded = False
            
            # Check if we can load a cached index
            if Path(INDEX_CACHE_PATH).exists() and Path(CORPUS_HASH_PATH).exists():
                try:
                    # Check if corpus hasn't changed
                    with open(CORPUS_HASH_PATH, 'r') as f:
                        cached_hash = json.load(f).get('hash', '')
                    
                    if cached_hash == corpus_hash:
                        print(f"📂 Loading cached search indices...")
                        if search_type == "multi_index":
                            multi_index.load(INDEX_CACHE_PATH)
                            index_loaded = True
                            print(f"✅ Cached search indices loaded successfully")
                        else:
                            # For simple search, we still need to build
                            print(f"⚠️ Simple search doesn't support caching, building new index...")
                except Exception as e:
                    print(f"⚠️ Failed to load cached index: {e}")
                    print(f"📍 Will build new index...")
            
            # Build new index if needed
            if not index_loaded:
                print(f"🔍 Building search indices for {len(quotes_df):,} passages...")
                
                # Convert to list with progress bar
                text_list = quotes_df['text'].tolist()
                
                if search_type == "multi_index":
                    # Build index with metadata
                    metadata_list = quotes_df.to_dict('records')
                    multi_index.fit(text_list, metadata_list)
                    
                    # Save the index and corpus hash
                    try:
                        print(f"💾 Saving search indices to cache...")
                        Path(INDEX_CACHE_PATH).parent.mkdir(parents=True, exist_ok=True)
                        multi_index.save(INDEX_CACHE_PATH)
                        
                        # Save corpus hash
                        with open(CORPUS_HASH_PATH, 'w') as f:
                            json.dump({'hash': corpus_hash, 'timestamp': str(pd.Timestamp.now())}, f)
                        
                        print(f"✅ Search indices cached successfully")
                    except Exception as e:
                        print(f"⚠️ Failed to save index cache: {e}")
                else:
                    multi_index.build_index(text_list)
                    
            print(f"✅ Search system initialized ({search_type})")
        else:
            print("❌ Search system not available")
            return False
        
        # Initialize Gemini client (optional)
        if gemini_available:
            try:
                gemini_client = GeminiClient()
                print("✅ Gemini client initialized")
            except Exception as e:
                print(f"⚠️ Gemini client not available: {e}")
                gemini_client = None
        else:
            print("⚠️ Gemini client not available")
            gemini_client = None
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading components: {e}")
        traceback.print_exc()
        return False

@app.route('/')
def index():
    """Main search page."""
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    """Search endpoint."""
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query cannot be empty'}), 400
        
        if multi_index is None or quotes_df is None:
            return jsonify({'error': 'Search system not initialized'}), 500
        
        # Normalize query
        if normalizer:
            try:
                normalized_query = normalizer.normalize_text(query)
            except Exception as e:
                print(f"Normalization error: {e}")
                normalized_query = query.lower().strip()
        else:
            normalized_query = query.lower().strip()
        
        # Retrieve top 5 results
        try:
            results = multi_index.search(normalized_query, top_k=5)
            print(f"Search results for '{normalized_query}': {len(results)} found")
        except Exception as e:
            print(f"Search error: {e}")
            return jsonify({'error': f'Search execution failed: {str(e)}'}), 500
        
        # Format results
        search_results = []
        for rank, result in enumerate(results, 1):
            if isinstance(result, dict):
                # MultiIndex returns dict format
                doc_idx = result.get('doc_id', 0)
                score = result.get('score', 0.0)
                text = result.get('text', '')
                metadata = result.get('metadata', {})
            else:
                # Simple search returns tuple format
                doc_idx, score = result
                if doc_idx < len(quotes_df):
                    text = str(quotes_df.iloc[doc_idx]['text'])
                    metadata = quotes_df.iloc[doc_idx].to_dict()
                else:
                    continue
            
            # Extract and enhance metadata
            if metadata_available and isinstance(metadata, dict):
                source_file = metadata.get('source_file', '')
                category_raw = metadata.get('category', '')
                
                # Use metadata mapper to get proper work info
                enhanced_metadata = metadata_mapper.extract_metadata(source_file, category_raw)
                work = enhanced_metadata.get('work', 'Unknown')
                author = enhanced_metadata.get('author', 'Unknown')
                category = enhanced_metadata.get('category', 'Unknown')
            else:
                work = metadata.get('work', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
                category = metadata.get('category', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
                author = metadata.get('author', 'Unknown') if isinstance(metadata, dict) else 'Unknown'
            
            formatted_result = {
                'rank': rank,
                'text': text,
                'work': work,
                'category': category,
                'score': float(score),
                'author': author
            }
            search_results.append(formatted_result)
        
        # Optional: Use Gemini for re-ranking (if available)
        if gemini_client and len(search_results) > 1:
            try:
                passages = [r['text'] for r in search_results]
                llm_scores = gemini_client.filter_relevant_passages(query, passages)
                
                # Update scores with LLM evaluation
                for i, llm_eval in enumerate(llm_scores):
                    if i < len(search_results):
                        search_results[i]['llm_score'] = llm_eval.get('relevance_score', 5)
                        search_results[i]['llm_explanation'] = llm_eval.get('explanation', '')
                
                # Re-sort by LLM score
                search_results.sort(key=lambda x: x.get('llm_score', x['score']), reverse=True)
                
            except Exception as e:
                print(f"LLM re-ranking failed: {e}")
        
        return jsonify({
            'query': query,
            'normalized_query': normalized_query,
            'results': search_results,
            'total_found': len(search_results),
            'llm_enhanced': gemini_client is not None
        })
        
    except Exception as e:
        print(f"Search error: {e}")
        traceback.print_exc()
        return jsonify({'error': f'Search failed: {str(e)}'}), 500

@app.route('/health')
def health():
    """Health check endpoint."""
    status = {
        'quotes_loaded': quotes_df is not None,
        'multi_index_ready': multi_index is not None,
        'normalizer_ready': normalizer is not None,
        'gemini_available': gemini_client is not None,
        'total_quotes': len(quotes_df) if quotes_df is not None else 0
    }
    return jsonify(status)

@app.route('/stats')
def stats():
    """Dataset statistics."""
    if quotes_df is None:
        return jsonify({'error': 'Dataset not loaded'}), 500
    
    stats = {
        'total_quotes': len(quotes_df),
        'works': quotes_df['work'].value_counts().to_dict() if 'work' in quotes_df.columns else {},
        'categories': quotes_df['category'].value_counts().to_dict() if 'category' in quotes_df.columns else {},
        'avg_length': quotes_df['text'].str.len().mean() if 'text' in quotes_df.columns else 0
    }
    return jsonify(stats)

if __name__ == '__main__':
    print("🚀 Starting ULLU Sanskrit Quote Search")
    print("=" * 50)
    
    # Load components
    if load_components():
        print("\n✅ All components loaded successfully!")
        print("🌐 Starting web server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    else:
        print("\n❌ Failed to load components. Please check your setup.")
        print("Make sure you have:")
        print("  - data/sanskrit_quotes.csv")
        print("  - All dependencies installed")
        print("  - .env file with API keys (optional)")