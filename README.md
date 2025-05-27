# ULLU
## Sanskrit Quote Retrieval Pipeline

**ULLU** is a 3-stage retrieval system for identifying the exact source of Sanskrit quotes from classical Indian literature.

Given an arbitrary Sanskrit quote (in IAST, HK, or Devanāgarī), ULLU identifies its source text (genre, work, book/chapter, verse).

## Architecture

### Stage 1: Coarse Retrieval (Recall@100 ≥ 95%)
- **Multi-index approach**: BM25 (lemmas), Character 3-grams, Dense embeddings (Sentence Transformers/TF-IDF)
- **Partial text matching**: Sliding window approach for substring queries
- **Sanskrit preprocessing**: Script normalization, Sandhi splitting, Morphological analysis
- **Synonym expansion**: Deity epithets and common terms
- **Knowledge Graph enhancement**: Entity extraction, relationship-based query expansion
- **Goal**: High recall to ensure ground truth is in top-100 candidates

### Stage 2: Fine-grained Ranking (Precision@1 maximized)
- **Cross-encoder model**: Random Forest with Sanskrit-specific features
- **Feature extraction**: Lexical overlap, semantic similarity, positional features
- **Hard negatives**: Adjacent ślokas, same chapter, same genre
- **Goal**: Maximize precision@1 through learned ranking

### Stage 3: LLM-based Filtering & Re-ranking
- **Gemini Integration**: Google's Gemini API for semantic relevance scoring
- **Re-ranking**: LLM-based passage relevance evaluation (1-10 scale)
- **Metadata extraction**: Work identification, genre classification, verse references
- **Fallback**: Rule-based filter using content patterns and citations
- **Output**: Final source identification with bibliographic details and relevance explanations

## Dataset

- **~2M passages** from Sanskrit digital corpus (full version available)
- **190K curated quotes** in the default dataset
- **Sources**: Purāṇas, Epics, Upaniṣads, Śāstras, Buddhist texts, Kāvya literature
- **Formats**: IAST transliteration with comprehensive metadata (work, author, category)

## Performance

### Current Results:
- **Stage 1 Recall@100**: 95%+ (including partial text queries)
- **Cross-encoder Performance**: 98.5% accuracy on test set
- **Partial Text Matching**: Successfully handles substring queries
- **Knowledge Graph**: Improves ranking for entity-related queries by 15-20%
- **LLM Re-ranking**: Gemini integration provides semantic relevance scoring
- **Index Loading Time**: <5 seconds with caching (vs 2-3 minutes without)

### Example Query:
```
Query: "devī cakre bhayākulā"

Result:
✓ Match: True (Confidence: 1.000)
Text: tataḥ sā capalāṃ dṛṣṭiṃ devī cakre bhayākulā /
      vilolitadṛśaṃ dṛṣṭvā punarāha ca tāṃ raviḥ // MarkP_77.5 //
Source: Mārkaṇḍeya Purāṇa 77.5
```

### KG-Enhanced Example:
```
Query: "kṛṣṇa arjuna"

Without KG: Generic passages ranked higher
With KG: Bhagavadgītā passages promoted (KG score: 0.214)
```

### Partial Text Example:
```
Full verse: "satyaṃ brūhi sutaḥ kasya somasyātha bṛhaspateḥ"
Partial query: "somasyātha bṛhaspateḥ"

Result: ✓ Successfully retrieved full verse using sliding window approach
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/ullu.git
cd ullu

# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env
# Edit .env and add your GEMINI_API_KEY

# Download data (if not included)
# Place sanskrit_corpus.csv or sanskrit_quotes.csv in data/
```

## Quick Start

### Run Web Interface
```bash
python3 app.py
# Visit http://localhost:5000
# First run will build indices (2-3 minutes)
# Subsequent runs use cache (<5 seconds)
```

### Run Command-line Demo
```bash
python3 demo.py
```

### Test Components
```bash
# Test KG system
python3 test_kg_system.py

# Test partial matching
python3 test_partial_matching.py

# Test Gemini integration
python3 test_gemini_search.py
```

### Build Full Pipeline
```bash
# Process corpus
python3 src/data_processing.py

# Build indices and models
python3 src/retrieval/simple_stage1.py
python3 src/ranking/cross_encoder.py

# Run complete pipeline
python3 src/pipeline.py
```

## Project Structure

```
ullu/
├── src/
│   ├── preprocessing/        # Text normalization, sandhi splitting
│   ├── indexing/            # Multi-index system
│   │   ├── multi_index.py   # Main indexing with KG & partial matching
│   │   └── partial_match_index.py  # Sliding window for substrings
│   ├── retrieval/           # Stage 1 coarse retrieval
│   ├── ranking/             # Stage 2 cross-encoder ranking
│   │   ├── cross_encoder.py # Random Forest ranker
│   │   └── xgboost_cross_encoder.py  # XGBoost alternative
│   ├── filtering/           # Stage 3 LLM-based filtering
│   ├── knowledge_graph/     # Entity extraction & KG enhancement
│   │   ├── entity_extractor.py  # Sanskrit entity recognition
│   │   ├── knowledge_base.py    # KG relationships & ontology
│   │   └── kg_enhancer.py       # Query expansion & reranking
│   ├── llm/                 # LLM integration
│   │   └── gemini_client.py # Google Gemini API client
│   └── pipeline.py          # End-to-end system
├── data/
│   ├── 1_sanskr/           # Sanskrit corpus (HTML/XML)
│   ├── sanskrit_corpus.csv # Full 2M passage dataset
│   ├── sanskrit_quotes.csv # Curated 190K quotes
│   ├── cached_multi_index.pkl  # Cached embeddings
│   └── *.pkl               # Trained models
├── app.py                  # Flask web interface
├── demo.py                 # Command-line demonstration
├── templates/              # Web UI templates
├── tests/                  # Test scripts
└── requirements.txt        # Dependencies
```

## Technical Details

### Sanskrit-specific Preprocessing
- **Script detection**: Auto-detect Devanāgarī, IAST, HK, ITRANS, SLP1
- **Normalization**: Unicode NFC, vedic accent removal
- **ASCII indexing**: Diacritic removal for fuzzy matching
- **Compound analysis**: Rule-based sandhi splitting

### Multi-index Retrieval
- **BM25**: Tuned for lemmatized Sanskrit (k1=1.2, b=0.75)
- **Character n-grams**: 3-gram Jaccard similarity for OCR robustness
- **Dense embeddings**: Sentence Transformers (GPU accelerated) with TF-IDF fallback
- **Partial matching**: Sliding window approach (window_size=10, overlap=5)
- **Score fusion**: Weighted combination (BM25: 0.5, n-gram: 0.3, dense: 0.2)
- **Index caching**: Automatic embedding cache to avoid recalculation
- **GPU support**: Automatic detection and usage for faster embeddings

### Cross-encoder Features
- **Lexical**: Word overlap, Jaccard similarity, length ratios
- **Semantic**: TF-IDF cosine similarity, character n-gram overlap
- **Positional**: Exact match detection, longest common subsequence
- **Model**: Random Forest (100 trees, max_depth=10)

### Knowledge Graph Enhancement
- **Entity extraction**: Deities, sages, places, concepts, works
- **Relationship mapping**: Avatar_of, consort_of, student_of, etc.
- **Query expansion**: Automatic expansion using entity relationships
- **Semantic filtering**: KG path-based validation and scoring
- **Work associations**: Entity-work co-occurrence analysis

### LLM Integration
- **Gemini API**: Google's Gemini-1.5-flash model for relevance scoring
- **Structured prompts**: JSON-based evaluation format
- **Relevance scoring**: 1-10 scale with explanations
- **Metadata extraction**: Work names, verse references, genre classification
- **Fallback handling**: Graceful degradation if API unavailable

## Dependencies

- **Core**: pandas, numpy, scikit-learn, xgboost
- **Sanskrit**: indic-transliteration (optional)
- **ML**: sentence-transformers, torch (GPU support)
- **LLM**: google-generativeai (Gemini API)
- **Web**: flask, requests, beautifulsoup4
- **Viz**: matplotlib, seaborn
- **Storage**: pickle (for index caching)
- **Config**: python-dotenv (for API keys)

## Evaluation Framework

- **Recall@k**: Stage 1 candidate retrieval effectiveness
- **Precision@1**: Stage 2 ranking accuracy
- **F1**: Stage 3 binary classification performance
- **End-to-end**: Quote→source identification success rate

## Key Features

1. **Knowledge Graph Integration**: ✅ 
   - Entity extraction (deities, sages, places, concepts, works)
   - Relationship-based query expansion
   - Semantic path-based scoring

2. **Partial Text Matching**: ✅
   - Sliding window approach for substring queries
   - Cross-window boundary detection
   - Configurable window size and overlap

3. **Index Caching**: ✅
   - Automatic embedding cache (5-second load vs 2-3 minutes)
   - Corpus hash validation
   - Intelligent cache invalidation

4. **LLM Integration**: ✅
   - Google Gemini API for semantic scoring
   - Structured JSON evaluation format
   - Graceful fallback handling

5. **Web Interface**: ✅
   - Interactive Flask-based UI
   - Real-time search results
   - Entity and relevance visualization

## Configuration

### Environment Setup
Create a `.env` file with:
```bash
GEMINI_API_KEY=your_api_key_here
```

### Index Configuration
```python
config = {
    'use_kg': True,          # Enable Knowledge Graph
    'use_partial': True,     # Enable partial matching
    'window_size': 10,       # Partial match window size
    'overlap': 5            # Window overlap
}
```

## Future Enhancements

1. **Advanced Morphology**: Integration with UoH Sanskrit tagger
2. **Additional LLMs**: GPT-4/Claude integration options
3. **Multilingual**: Support for regional script variants
4. **Real-time API**: RESTful API for programmatic access
5. **Expanded KG**: More entities, verse-level associations
6. **Advanced Sandhi**: ML-based sandhi splitting

## Research Applications

- **Digital Humanities**: Source attribution for Sanskrit texts
- **Scholarly Research**: Citation verification and analysis
- **Educational Tools**: Interactive Sanskrit learning platforms
- **Cultural Heritage**: Digitization and cross-referencing projects

---

Built as part of the Sanskrit Quote Retrieval Challenge.
Designed for scholars, students, and enthusiasts of Sanskrit literature.

🕉️ *"यत्र विद्वांसस्तत्र देवाः"* - Where there are scholars, there are the gods.
