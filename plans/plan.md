**Detailed Design Plan: Sanskrit Quote Retrieval Pipeline**

**1. Overview**
- **Objective:** Given an arbitrary Sanskrit quote (IAST, HK, or Devanāgarī), identify its source text (genre, work, book/chapter, verse).
- **Architecture:** 3-stage pipeline with Sanskrit-specific preprocessing and indexing enhancements.

---

**2. Preprocessing Layer**
Standardize input and corpus before retrieval:

- **Script & Diacritic Normalization**
  - Convert all texts and queries to Unicode IAST NFC.
  - Maintain a stripped-ASCII index for fuzzy matching.

- **Sandhi Splitting & Compound Segmentation**
  - Run rule-based Sandhi splitter (e.g. Sanskrit Heritage API) on both query and corpus.
  - Decompose compounds into constituent stems.

- **Morphological Analysis & Lemmatization**
  - Use a morphological analyzer (e.g. UoH Sanskrit tagger) to annotate each token with lemma+features.
  - Index both inflected forms and lemmas.

---

**3. Indexing Strategy**
Optimize for high recall and precision:

- **Verse-Level Index Units**
  - Index each śloka or sentence as one unit, preserving metadata (work, book/chapter, verse number).

- **Multi-Index Approach**
  1. **BM25 on lemmas + stems** (lexical recall)
  2. **Character 3-gram index** on Sandhi-split tokens (fuzzy match + OCR robustness)
  3. **Dense dual-encoder embeddings** over lemmas (optional, for paraphrased queries)

- **Synonym & Epithet Expansion**
  - At query time, expand known epithets/variants (e.g. rādhā ↔ śrīrādhā) via small dictionary.

- **Knowledge-Graph-Based Enhancement**
  - Leverage a structured ontology of Sanskrit works, authors, and concepts:
    - Entity Extraction & Linking
      - Use NLP pipelines to detect named entities (e.g. deities, sages, places) in quotes.
      - Link entities against a curated KG of works and characters (e.g. Kavyamala ontology).
    - Graph-Driven Expansion
      - From extracted entities, traverse KG relations to identify candidate works or chapters (e.g. all verses mentioning Viṣṇu in Bhagavadgītā).
    - Augment retrieval candidates with passages connected via entity or concept relations.
    - Semantic Filtering via KG Paths
      - At filtering stage, validate that the top-ranked candidate’s metadata matches expected KG relations (e.g. quote mentions Draupadī; candidate from Mahābhārata).
      - Compute path-based features (entity co-occurrence, relation distance) as additional signals for ranking/filtering.

---

**4. Stage 1: Retrieval (Coarse Candidates)**
- **Goal:** Recall@100 ≥ 95%
- **Process:**
  1. Normalize + split + lemmatize query.
  2. Run BM25 on lemma-index, fallback to 3-gram index for unmapped tokens.
  3. (Optionally) retrieve dense candidates via dual-encoder.
- **Output:** Top-K passage IDs with metadata.

---

**5. Stage 2: Ranking (Cross-Encoder)**
- **Goal:** Precision@1 maximized.
- **Model:** Cross-encoder fine-tuned on (quote, passage) pairs with hard negatives from same text.
- **Training Data:**
  - Positive: exact matches.
  - Hard negatives: adjacent ślokas, same chapter, same genre.
- **Inference:** Score Top-K candidates, sort descending.

---

**6. Stage 3: Filtering (LLM-Based Confirmation)**
- **Goal:** Binary accept/reject & granular label prediction (work, book, verse).
- **Approach:** Prompt-tune a pretrained LLM or fine-tune classification head:
  - **Input:** `[QUOTE] + [Top-1 passage text + metadata]`
  - **Output:**
    ```json
    {
      "match": "yes" | "no",
      "predicted_genre": "...",
      "predicted_work": "...",
      "predicted_book": "...",
      "predicted_verse": "..."
    }
    ```
- **Training:** Supervised on labeled triples, include near-miss negatives.

---

**7. Hard Negative Mining & Iteration**
- Automatically extract negatives at all three stages:
  - **Retrieval:** random verses from different works.
  - **Ranking:** neighbor verses.
  - **Filtering:** passages from same work but different chapter.
- **Loop:** monitor per-stage metrics and add new negatives for any failure modes.

---

**8. Evaluation & Metrics**
- **Stage 1:** Recall@100 on held-out set.
- **Stage 2:** MRR & Precision@1.
- **Stage 3:** Accuracy / F1 of match classifier.
- **End-to-End:** Fraction of quotes with correct final identification.

---

**9. Tech Stack & Tools**
- **Preprocessing:** Sanskrit Heritage API, UoH Sandhi splitter, stanza-sanskrit.
- **Indexing:** Elasticsearch / OpenSearch (lexical + n-gram indices).
- **Dense Retrieval:** SentenceTransformers DPR or similar.
- **Cross-Encoder:** Hugging Face Transformers (e.g. `bert-base-multilingual-cased`).
- **LLM Filtering:** LLaMA-based model or OpenAI GPT (with fine-tuning/classification).

---

**10. Next Steps**
1. Collect & annotate a balanced dataset of (quote → exact passage).
2. Implement preprocessing pipeline and index creation.
3. Prototype BM25 retrieval & measure recall.
4. Fine-tune cross-encoder; evaluate ranking.
5. Develop LLM-based filter classifier; integrate end-to-end.
6. Monitor metrics; refine hard-negative mining.

---
