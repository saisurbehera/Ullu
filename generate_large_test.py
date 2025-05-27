#!/usr/bin/env python3
"""
Generate Large Test Dataset for Cross-Encoder Evaluation
Creates 100K+ test pairs efficiently from the corpus.
"""

import pandas as pd
import numpy as np
import random
import logging
from pathlib import Path
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def generate_large_test_dataset(target_size: int = 100000):
    """Generate large test dataset efficiently."""
    
    print(f"🚀 Generating Large Test Dataset ({target_size:,} pairs)")
    print("=" * 50)
    
    # Load datasets
    quotes_path = "data/sanskrit_quotes.csv" 
    corpus_path = "data/sanskrit_corpus.csv"
    
    if Path(corpus_path).exists():
        print("📚 Loading large corpus...")
        corpus_df = pd.read_csv(corpus_path, dtype={'text': str})
        corpus_df = corpus_df.dropna(subset=['text'])
        print(f"   Large corpus: {len(corpus_df):,} passages")
    else:
        print("📚 Using quotes corpus...")
        corpus_df = pd.read_csv(quotes_path)
        print(f"   Quotes corpus: {len(corpus_df):,} passages")
    
    # Use larger sample for 100K dataset
    if len(corpus_df) > 200000:
        corpus_sample = corpus_df.sample(n=200000, random_state=42)
        print(f"📊 Sampled {len(corpus_sample):,} passages for 100K generation")
    else:
        corpus_sample = corpus_df
        print(f"📊 Using full corpus: {len(corpus_sample):,} passages")
    
    all_test_cases = []
    
    # 1. Exact matches (40% of dataset)
    print("\n🎯 Generating exact match cases...")
    exact_target = int(target_size * 0.4)
    
    sampled_for_exact = corpus_sample.sample(n=min(exact_target // 5, len(corpus_sample)), random_state=42)
    
    for idx, row in sampled_for_exact.iterrows():
        text = str(row['text'])
        words = text.split()
        
        if len(words) < 4:
            continue
        
        # Generate 5 different exact match queries per passage
        for ratio in [0.2, 0.35, 0.5, 0.65, 0.8]:
            query_len = max(2, int(len(words) * ratio))
            start_pos = random.randint(0, max(0, len(words) - query_len))
            query = ' '.join(words[start_pos:start_pos + query_len])
            
            if len(query.strip()) > 10:
                all_test_cases.append({
                    'query': query,
                    'passage': text,
                    'label': 1,
                    'test_type': 'exact_match',
                    'expected_score': 'very_high',
                    'difficulty': 'easy',
                    'work': row.get('work', 'unknown'),
                    'category': row.get('category', 'unknown')
                })
        
        if len(all_test_cases) >= exact_target:
            break
    
    print(f"   Generated {len(all_test_cases):,} exact match cases")
    
    # 2. Partial matches (25% of dataset)
    print("\n🔀 Generating partial match cases...")
    partial_target = int(target_size * 0.25)
    
    sampled_for_partial = corpus_sample.sample(n=min(partial_target // 4, len(corpus_sample)), random_state=43)
    
    for idx, row in sampled_for_partial.iterrows():
        text = str(row['text'])
        words = text.split()
        
        if len(words) < 6:
            continue
        
        # Generate 4 partial match variants
        for overlap_ratio in [0.3, 0.5, 0.7, 0.9]:
            query_length = 6
            num_overlapping = max(2, int(query_length * overlap_ratio))
            
            # Take some words from passage
            overlapping_words = random.sample(words, min(num_overlapping, len(words)))
            
            # Add random words
            remaining_slots = query_length - len(overlapping_words)
            if remaining_slots > 0:
                other_passage = corpus_sample.sample(1).iloc[0]
                other_words = str(other_passage['text']).split()
                random_words = random.sample(other_words, min(remaining_slots, len(other_words)))
                overlapping_words.extend(random_words)
            
            random.shuffle(overlapping_words)
            query = ' '.join(overlapping_words[:query_length])
            
            if len(query.strip()) > 10:
                difficulty = 'medium' if overlap_ratio >= 0.5 else 'hard'
                
                all_test_cases.append({
                    'query': query,
                    'passage': text,
                    'label': 1,
                    'test_type': 'partial_match',
                    'expected_score': 'medium',
                    'difficulty': difficulty,
                    'work': row.get('work', 'unknown'),
                    'category': row.get('category', 'unknown'),
                    'overlap_ratio': overlap_ratio
                })
        
        if len(all_test_cases) >= exact_target + partial_target:
            break
    
    current_positives = len(all_test_cases)
    print(f"   Generated {current_positives - exact_target:,} partial match cases")
    
    # 3. Semantic matches (10% of dataset)
    print("\n🔗 Generating semantic match cases...")
    semantic_target = int(target_size * 0.1)
    
    # Sanskrit synonyms
    synonyms = {
        'viṣṇu': ['hari', 'nārāyaṇa', 'keśava', 'kṛṣṇa'],
        'śiva': ['maheśvara', 'śaṅkara', 'rudra', 'hara'],
        'dharma': ['nyāya', 'satya', 'rita'],
        'mokṣa': ['mukti', 'kaivalya', 'nirvāṇa'],
        'rāja': ['nṛpa', 'bhūpati', 'mahīpati']
    }
    
    semantic_count = 0
    for original, synonym_list in synonyms.items():
        matching_passages = corpus_sample[
            corpus_sample['text'].str.contains(original, case=False, na=False)
        ]
        
        for _, row in matching_passages.sample(n=min(100, len(matching_passages))).iterrows():
            text = str(row['text'])
            
            for synonym in synonym_list[:2]:
                # Simple substitution query
                words = text.lower().split()[:6]
                query_words = [synonym if original in word else word for word in words]
                query = ' '.join(query_words)
                
                if len(query.strip()) > 10:
                    all_test_cases.append({
                        'query': query,
                        'passage': text,
                        'label': 1,
                        'test_type': 'semantic_match',
                        'expected_score': 'medium-high',
                        'difficulty': 'hard',
                        'work': row.get('work', 'unknown'),
                        'category': row.get('category', 'unknown'),
                        'original_term': original,
                        'synonym_used': synonym
                    })
                    
                    semantic_count += 1
                    if semantic_count >= semantic_target:
                        break
            
            if semantic_count >= semantic_target:
                break
        
        if semantic_count >= semantic_target:
            break
    
    print(f"   Generated {semantic_count:,} semantic match cases")
    
    # 4. Negative cases (25% of dataset)
    print("\n❌ Generating negative cases...")
    positive_cases = len(all_test_cases)
    negative_target = int(target_size * 0.25)
    
    # Sample queries from positive cases
    queries_for_negatives = random.sample(all_test_cases, min(negative_target // 3, len(all_test_cases)))
    
    for pos_case in queries_for_negatives:
        query = pos_case['query']
        pos_work = pos_case['work']
        
        # Generate 3 types of negatives per query
        for neg_type in ['random', 'same_category', 'similar_length']:
            try:
                if neg_type == 'random':
                    neg_passage = corpus_sample.sample(1).iloc[0]
                    difficulty = 'easy'
                    
                elif neg_type == 'same_category':
                    pos_category = pos_case['category']
                    same_cat = corpus_sample[corpus_sample['category'] == pos_category]
                    if len(same_cat) > 1:
                        neg_passage = same_cat.sample(1).iloc[0]
                        difficulty = 'medium'
                    else:
                        continue
                        
                elif neg_type == 'similar_length':
                    pos_length = len(pos_case['passage'])
                    # Find passages with similar length (±20%)
                    length_range = pos_length * 0.2
                    similar_length = corpus_sample[
                        (corpus_sample['text'].str.len() >= pos_length - length_range) &
                        (corpus_sample['text'].str.len() <= pos_length + length_range)
                    ]
                    if len(similar_length) > 1:
                        neg_passage = similar_length.sample(1).iloc[0]
                        difficulty = 'medium'
                    else:
                        continue
                
                if str(neg_passage['text']) != pos_case['passage']:
                    all_test_cases.append({
                        'query': query,
                        'passage': str(neg_passage['text']),
                        'label': 0,
                        'test_type': f'negative_{neg_type}',
                        'expected_score': 'very_low',
                        'difficulty': difficulty,
                        'work': neg_passage.get('work', 'unknown'),
                        'category': neg_passage.get('category', 'unknown'),
                        'negative_type': neg_type
                    })
                    
            except Exception as e:
                continue
        
        if len(all_test_cases) >= target_size:
            break
    
    negative_count = len(all_test_cases) - positive_cases
    print(f"   Generated {negative_count:,} negative cases")
    
    # Create final dataset
    random.shuffle(all_test_cases)
    test_df = pd.DataFrame(all_test_cases[:target_size])
    test_df['test_id'] = range(len(test_df))
    
    return test_df

def main():
    """Generate large test dataset."""
    start_time = time.time()
    
    # Generate dataset
    test_df = generate_large_test_dataset(target_size=100000)
    
    # Save dataset
    output_path = "data/large_cross_encoder_test.csv"
    test_df.to_csv(output_path, index=False)
    
    generation_time = time.time() - start_time
    
    print(f"\n📊 LARGE TEST DATASET COMPLETE")
    print("=" * 40)
    print(f"Total pairs: {len(test_df):,}")
    print(f"Positive pairs: {len(test_df[test_df['label'] == 1]):,}")
    print(f"Negative pairs: {len(test_df[test_df['label'] == 0]):,}")
    print(f"Generation time: {generation_time:.1f}s")
    
    print(f"\nTest type distribution:")
    for test_type, count in test_df['test_type'].value_counts().items():
        percentage = count / len(test_df) * 100
        print(f"  {test_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nDifficulty distribution:")
    for difficulty, count in test_df['difficulty'].value_counts().items():
        percentage = count / len(test_df) * 100
        print(f"  {difficulty}: {count:,} ({percentage:.1f}%)")
    
    file_size = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n💾 Dataset saved to: {output_path}")
    print(f"File size: {file_size:.1f} MB")
    
    print(f"\n🧪 Next Steps:")
    print(f"   Test with: python3 test_cross_encoders.py")
    print(f"   Full eval: python3 src/evaluation/run_cross_encoder_tests.py")

if __name__ == "__main__":
    main()