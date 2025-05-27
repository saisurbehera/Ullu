import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import random
import re
from pathlib import Path
from collections import defaultdict
import itertools

logger = logging.getLogger(__name__)

class LargeTestDatasetGenerator:
    """Generate very large, comprehensive test datasets for cross-encoder evaluation."""
    
    def __init__(self, corpus_path: str):
        self.corpus_df = pd.read_csv(corpus_path)
        self.large_corpus_path = "/home/sai/Desktop/ullu/data/sanskrit_corpus.csv"
        
        # Load large corpus if available
        if Path(self.large_corpus_path).exists():
            self.large_corpus_df = pd.read_csv(self.large_corpus_path)
            logger.info(f"Loaded large corpus: {len(self.large_corpus_df)} passages")
        else:
            self.large_corpus_df = self.corpus_df
            logger.info(f"Using quotes corpus: {len(self.corpus_df)} passages")
        
        self.test_cases = []
        self._index_by_attributes()
    
    def _index_by_attributes(self):
        """Index passages by various attributes for efficient sampling."""
        self.by_work = defaultdict(list)
        self.by_category = defaultdict(list)
        self.by_length = defaultdict(list)
        self.by_word_count = defaultdict(list)
        
        for idx, row in self.large_corpus_df.iterrows():
            text = str(row.get('text', ''))
            work = row.get('work', 'unknown')
            category = row.get('category', 'unknown')
            
            passage_data = {
                'idx': idx,
                'text': text,
                'work': work,
                'category': category,
                'length': len(text),
                'word_count': len(text.split())
            }
            
            # Index by work
            self.by_work[work].append(passage_data)
            
            # Index by category
            self.by_category[category].append(passage_data)
            
            # Index by length buckets
            length_bucket = (len(text) // 50) * 50  # 50-char buckets
            self.by_length[length_bucket].append(passage_data)
            
            # Index by word count buckets
            word_bucket = (len(text.split()) // 5) * 5  # 5-word buckets
            self.by_word_count[word_bucket].append(passage_data)
    
    def generate_exhaustive_exact_matches(self, target_count: int = 10000) -> List[Dict]:
        """Generate exhaustive exact match test cases."""
        test_cases = []
        
        # Use all available passages
        all_passages = self.large_corpus_df.sample(frac=1, random_state=42)  # Shuffle
        
        for idx, row in all_passages.iterrows():
            if len(test_cases) >= target_count:
                break
                
            text = str(row.get('text', ''))
            words = text.split()
            
            if len(words) < 3:
                continue
            
            # Generate multiple query variants per passage
            query_variants = []
            
            # Different substring lengths
            for ratio in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                query_len = max(2, int(len(words) * ratio))
                
                # Multiple starting positions
                for start_pos in range(0, max(1, len(words) - query_len + 1), max(1, query_len // 2)):
                    if start_pos + query_len <= len(words):
                        query = ' '.join(words[start_pos:start_pos + query_len])
                        if len(query.strip()) > 10:  # Minimum query length
                            query_variants.append(query)
            
            # Add all variants
            for query in query_variants:
                test_cases.append({
                    'query': query,
                    'passage': text,
                    'label': 1,
                    'test_type': 'exact_match',
                    'expected_score': 'very_high',
                    'difficulty': 'easy',
                    'work': row.get('work', 'unknown'),
                    'category': row.get('category', 'unknown'),
                    'substring_ratio': len(query.split()) / len(words),
                    'passage_source': 'large_corpus'
                })
                
                if len(test_cases) >= target_count:
                    break
        
        logger.info(f"Generated {len(test_cases)} exact match test cases")
        return test_cases
    
    def generate_comprehensive_partial_matches(self, target_count: int = 15000) -> List[Dict]:
        """Generate comprehensive partial match test cases."""
        test_cases = []
        
        # Use large corpus for diversity
        sampled_passages = self.large_corpus_df.sample(n=min(target_count // 3, len(self.large_corpus_df)), random_state=43)
        
        for idx, row in sampled_passages.iterrows():
            if len(test_cases) >= target_count:
                break
                
            text = str(row.get('text', ''))
            words = text.split()
            
            if len(words) < 6:
                continue
            
            # Generate multiple partial match variants
            for overlap_ratio in [0.2, 0.4, 0.6, 0.8]:
                for query_length in [4, 6, 8, 10]:
                    if query_length >= len(words):
                        continue
                    
                    num_overlapping = max(1, int(query_length * overlap_ratio))
                    num_non_overlapping = query_length - num_overlapping
                    
                    # Sample overlapping words from passage
                    overlapping_words = random.sample(words, min(num_overlapping, len(words)))
                    
                    # Sample non-overlapping words from other passages
                    non_overlapping_words = []
                    for _ in range(num_non_overlapping):
                        other_passage = self.large_corpus_df.sample(1).iloc[0]
                        other_words = str(other_passage.get('text', '')).split()
                        if other_words:
                            word = random.choice(other_words)
                            if word not in overlapping_words:
                                non_overlapping_words.append(word)
                    
                    # Combine and shuffle
                    all_query_words = overlapping_words + non_overlapping_words
                    random.shuffle(all_query_words)
                    query = ' '.join(all_query_words[:query_length])
                    
                    if len(query.strip()) > 10:
                        # Determine difficulty
                        if overlap_ratio >= 0.6:
                            difficulty = 'medium'
                            expected_score = 'medium-high'
                        elif overlap_ratio >= 0.4:
                            difficulty = 'medium'
                            expected_score = 'medium'
                        else:
                            difficulty = 'hard'
                            expected_score = 'low-medium'
                        
                        test_cases.append({
                            'query': query,
                            'passage': text,
                            'label': 1,
                            'test_type': 'partial_match',
                            'expected_score': expected_score,
                            'difficulty': difficulty,
                            'work': row.get('work', 'unknown'),
                            'category': row.get('category', 'unknown'),
                            'overlap_ratio': overlap_ratio,
                            'query_length': query_length,
                            'passage_source': 'large_corpus'
                        })
                        
                        if len(test_cases) >= target_count:
                            break
                
                if len(test_cases) >= target_count:
                    break
            
            if len(test_cases) >= target_count:
                break
        
        logger.info(f"Generated {len(test_cases)} partial match test cases")
        return test_cases
    
    def generate_extended_semantic_matches(self, target_count: int = 8000) -> List[Dict]:
        """Generate extended semantic match test cases with comprehensive Sanskrit knowledge."""
        test_cases = []
        
        # Extended Sanskrit synonyms and related terms
        semantic_groups = {
            # Deities and epithets
            'viṣṇu': ['hari', 'nārāyaṇa', 'keśava', 'madhusūdana', 'govinda', 'kṛṣṇa', 'vāsudeva', 'janārdana', 'acyuta', 'ananta'],
            'śiva': ['maheśvara', 'śaṅkara', 'rudra', 'īśa', 'hara', 'śambhu', 'tryambaka', 'neelkantha', 'natarāja', 'bhairava'],
            'brahmā': ['pitāmaha', 'svayambhū', 'prajāpati', 'virañci', 'caturmukha', 'kamalāsana', 'vedhas', 'dhātṛ'],
            'devī': ['ambā', 'umā', 'pārvatī', 'śakti', 'bhagavatī', 'mātā', 'durgā', 'kālī', 'lakṣmī', 'sarasvatī'],
            'indra': ['śakra', 'maghavan', 'vāsava', 'surendra', 'devendra', 'purandara', 'vajrin'],
            'sūrya': ['āditya', 'ravi', 'bhāskara', 'divākara', 'savitṛ', 'mitra', 'arka'],
            
            # Philosophical concepts
            'dharma': ['nyāya', 'satya', 'rita', 'yuga', 'svadharma', 'rājadharma'],
            'karma': ['kriyā', 'cikīrṣā', 'pravṛtti', 'ceṣṭā'],
            'mokṣa': ['mukti', 'nirvāṇa', 'kaivalya', 'apavarga', 'liberation', 'videhamukti'],
            'yoga': ['yuj', 'samādhi', 'dhyāna', 'ekāgratā'],
            'jñāna': ['vidyā', 'vijñāna', 'prajñā', 'bodha', 'cit'],
            'bhakti': ['prema', 'śraddhā', 'anurakti', 'bhāva'],
            
            # Social/political terms
            'rāja': ['nṛpa', 'bhūpati', 'bhūpāla', 'mahīpati', 'kṣatriya', 'rājā', 'narendrā'],
            'guru': ['ācārya', 'śikṣaka', 'upādhyāya', 'preceptor'],
            'śiṣya': ['chātra', 'vidyārthī', 'antevāsin'],
            
            # Religious terms
            'mantra': ['gāyatrī', 'ṛk', 'yajus', 'sāman', 'japa'],
            'yajña': ['homa', 'havana', 'agnihotra', 'sacrifice'],
            'āśrama': ['gṛhastha', 'brahmacārī', 'vānaprastha', 'sannyāsa'],
            'varna': ['brāhmaṇa', 'kṣatriya', 'vaiśya', 'śūdra'],
            
            # Cosmic/temporal
            'kāla': ['samaya', 'velā', 'yuga', 'kalpa'],
            'loka': ['bhuvana', 'jagat', 'viśva', 'brahmāṇḍa'],
            'prāṇa': ['vāyu', 'marut', 'pavana', 'anila']
        }
        
        # Process each semantic group
        for primary_term, related_terms in semantic_groups.items():
            # Find passages containing the primary term
            matching_passages = self.large_corpus_df[
                self.large_corpus_df['text'].str.contains(primary_term, case=False, na=False)
            ]
            
            if len(matching_passages) == 0:
                continue
            
            # Sample passages for this term
            num_passages = min(len(matching_passages), target_count // len(semantic_groups))
            sampled_passages = matching_passages.sample(n=num_passages, random_state=42)
            
            for _, passage_row in sampled_passages.iterrows():
                text = str(passage_row.get('text', ''))
                
                # Create queries using related terms
                for related_term in related_terms[:5]:  # Use first 5 related terms
                    # Method 1: Simple substitution
                    query_words = text.lower().split()[:8]  # Take first 8 words
                    substituted_query = []
                    
                    for word in query_words:
                        if primary_term in word:
                            substituted_query.append(word.replace(primary_term, related_term))
                        else:
                            substituted_query.append(word)
                    
                    if len(substituted_query) >= 3:
                        query = ' '.join(substituted_query)
                        
                        test_cases.append({
                            'query': query,
                            'passage': text,
                            'label': 1,
                            'test_type': 'semantic_match',
                            'expected_score': 'medium-high',
                            'difficulty': 'hard',
                            'work': passage_row.get('work', 'unknown'),
                            'category': passage_row.get('category', 'unknown'),
                            'primary_term': primary_term,
                            'related_term': related_term,
                            'semantic_group': 'substitution',
                            'passage_source': 'large_corpus'
                        })
                    
                    # Method 2: Context-based query
                    # Create query with related term in context
                    context_words = ['namaḥ', 'śrī', 'bhagavān', 'mahā', 'para']
                    for context in context_words[:2]:
                        context_query = f"{context} {related_term}"
                        
                        test_cases.append({
                            'query': context_query,
                            'passage': text,
                            'label': 1,
                            'test_type': 'semantic_context',
                            'expected_score': 'medium',
                            'difficulty': 'hard',
                            'work': passage_row.get('work', 'unknown'),
                            'category': passage_row.get('category', 'unknown'),
                            'primary_term': primary_term,
                            'related_term': related_term,
                            'semantic_group': 'context',
                            'passage_source': 'large_corpus'
                        })
                        
                        if len(test_cases) >= target_count:
                            break
                    
                    if len(test_cases) >= target_count:
                        break
                
                if len(test_cases) >= target_count:
                    break
            
            if len(test_cases) >= target_count:
                break
        
        logger.info(f"Generated {len(test_cases)} semantic match test cases")
        return test_cases
    
    def generate_massive_negative_dataset(self, positive_cases: List[Dict], 
                                        negatives_per_positive: int = 4) -> List[Dict]:
        """Generate massive negative dataset with various difficulty levels."""
        negative_cases = []
        
        total_target = len(positive_cases) * negatives_per_positive
        logger.info(f"Generating {total_target} negative test cases...")
        
        # Create different types of negatives
        negative_types = [
            ('random', 0.3),           # 30% completely random
            ('same_category', 0.25),   # 25% same category
            ('same_work', 0.2),        # 20% same work
            ('similar_length', 0.15),  # 15% similar length
            ('partial_overlap', 0.1)   # 10% partial word overlap (hard negatives)
        ]
        
        for pos_case in positive_cases:
            query = pos_case['query']
            pos_work = pos_case.get('work', 'unknown')
            pos_category = pos_case.get('category', 'unknown')
            pos_length = len(pos_case['passage'])
            
            cases_generated = 0
            
            for neg_type, ratio in negative_types:
                num_for_type = int(negatives_per_positive * ratio)
                if num_for_type == 0:
                    continue
                
                for _ in range(num_for_type):
                    if cases_generated >= negatives_per_positive:
                        break
                    
                    try:
                        if neg_type == 'random':
                            # Completely random passage
                            neg_passage = self.large_corpus_df.sample(1).iloc[0]
                            expected_score = 'very_low'
                            difficulty = 'easy'
                            
                        elif neg_type == 'same_category':
                            # Same category, different work
                            candidates = self.by_category.get(pos_category, [])
                            candidates = [c for c in candidates if c['work'] != pos_work]
                            if candidates:
                                neg_passage_data = random.choice(candidates)
                                neg_passage = pd.Series(neg_passage_data)
                                expected_score = 'low'
                                difficulty = 'medium'
                            else:
                                continue
                                
                        elif neg_type == 'same_work':
                            # Same work, different passage
                            candidates = self.by_work.get(pos_work, [])
                            candidates = [c for c in candidates if c['text'] != pos_case['passage']]
                            if candidates:
                                neg_passage_data = random.choice(candidates)
                                neg_passage = pd.Series(neg_passage_data)
                                expected_score = 'low-medium'
                                difficulty = 'hard'
                            else:
                                continue
                                
                        elif neg_type == 'similar_length':
                            # Similar length passage
                            length_bucket = (pos_length // 50) * 50
                            candidates = self.by_length.get(length_bucket, [])
                            if not candidates:
                                # Try nearby length buckets
                                for offset in [50, -50, 100, -100]:
                                    candidates = self.by_length.get(length_bucket + offset, [])
                                    if candidates:
                                        break
                            
                            if candidates:
                                candidates = [c for c in candidates if c['text'] != pos_case['passage']]
                                if candidates:
                                    neg_passage_data = random.choice(candidates)
                                    neg_passage = pd.Series(neg_passage_data)
                                    expected_score = 'low'
                                    difficulty = 'medium'
                                else:
                                    continue
                            else:
                                continue
                                
                        elif neg_type == 'partial_overlap':
                            # Hard negative: some word overlap but wrong passage
                            query_words = set(query.lower().split())
                            
                            # Find passages with some word overlap
                            candidates = []
                            for idx, row in self.large_corpus_df.sample(200).iterrows():
                                passage_text = str(row.get('text', ''))
                                passage_words = set(passage_text.lower().split())
                                overlap = len(query_words.intersection(passage_words))
                                
                                if overlap > 0 and overlap < len(query_words) * 0.8 and passage_text != pos_case['passage']:
                                    candidates.append(row)
                            
                            if candidates:
                                neg_passage = random.choice(candidates)
                                expected_score = 'low-medium'
                                difficulty = 'very_hard'
                            else:
                                continue
                        
                        # Create negative test case
                        negative_cases.append({
                            'query': query,
                            'passage': str(neg_passage.get('text', '')),
                            'label': 0,
                            'test_type': f'negative_{neg_type}',
                            'expected_score': expected_score,
                            'difficulty': difficulty,
                            'work': neg_passage.get('work', 'unknown'),
                            'category': neg_passage.get('category', 'unknown'),
                            'negative_type': neg_type,
                            'passage_source': 'large_corpus'
                        })
                        
                        cases_generated += 1
                        
                    except Exception as e:
                        logger.warning(f"Error generating {neg_type} negative: {e}")
                        continue
        
        logger.info(f"Generated {len(negative_cases)} negative test cases")
        return negative_cases
    
    def generate_massive_test_dataset(self, 
                                    target_size: int = 50000,
                                    save_path: Optional[str] = None) -> pd.DataFrame:
        """Generate massive comprehensive test dataset."""
        logger.info(f"Generating massive test dataset (target: {target_size:,} pairs)...")
        
        # Allocate sizes for different test types
        exact_match_target = int(target_size * 0.25)      # 25%
        partial_match_target = int(target_size * 0.35)    # 35%
        semantic_match_target = int(target_size * 0.15)   # 15%
        # Remaining 25% will be negatives
        
        # Generate positive test cases
        logger.info("Generating exact match test cases...")
        exact_matches = self.generate_exhaustive_exact_matches(exact_match_target)
        
        logger.info("Generating partial match test cases...")
        partial_matches = self.generate_comprehensive_partial_matches(partial_match_target)
        
        logger.info("Generating semantic match test cases...")
        semantic_matches = self.generate_extended_semantic_matches(semantic_match_target)
        
        # Combine positive cases
        all_positive_cases = exact_matches + partial_matches + semantic_matches
        logger.info(f"Total positive cases: {len(all_positive_cases):,}")
        
        # Generate negative cases
        logger.info("Generating massive negative dataset...")
        negative_cases = self.generate_massive_negative_dataset(
            all_positive_cases[:target_size//5],  # Use subset for negatives to manage size
            negatives_per_positive=4
        )
        
        # Combine all test cases
        all_test_cases = all_positive_cases + negative_cases
        
        # Shuffle and create DataFrame
        random.shuffle(all_test_cases)
        test_df = pd.DataFrame(all_test_cases)
        
        # Add metadata
        test_df['test_id'] = range(len(test_df))
        test_df['dataset_version'] = 'massive_v1'
        
        # Trim to target size if exceeded
        if len(test_df) > target_size:
            test_df = test_df.sample(n=target_size, random_state=42).reset_index(drop=True)
        
        logger.info(f"Final dataset size: {len(test_df):,}")
        logger.info(f"Positive cases: {len(test_df[test_df['label'] == 1]):,}")
        logger.info(f"Negative cases: {len(test_df[test_df['label'] == 0]):,}")
        
        # Show distributions
        logger.info("Test type distribution:")
        for test_type, count in test_df['test_type'].value_counts().items():
            logger.info(f"  {test_type}: {count:,}")
        
        logger.info("Difficulty distribution:")
        for difficulty, count in test_df['difficulty'].value_counts().items():
            logger.info(f"  {difficulty}: {count:,}")
        
        if save_path:
            test_df.to_csv(save_path, index=False)
            logger.info(f"Massive test dataset saved to {save_path}")
        
        return test_df

def main():
    """Generate massive test dataset."""
    logging.basicConfig(level=logging.INFO)
    
    corpus_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    output_path = "/home/sai/Desktop/ullu/data/massive_cross_encoder_test.csv"
    
    if not Path(corpus_path).exists():
        print("❌ Corpus not found. Run data processing first.")
        return
    
    print("🚀 MASSIVE Cross-Encoder Test Dataset Generation")
    print("=" * 60)
    
    generator = LargeTestDatasetGenerator(corpus_path)
    
    # Generate massive dataset (50K pairs)
    massive_df = generator.generate_massive_test_dataset(
        target_size=50000,
        save_path=output_path
    )
    
    print(f"\n📊 MASSIVE DATASET GENERATED")
    print("=" * 40)
    print(f"Total test pairs: {len(massive_df):,}")
    print(f"Positive pairs: {len(massive_df[massive_df['label'] == 1]):,}")
    print(f"Negative pairs: {len(massive_df[massive_df['label'] == 0]):,}")
    
    print(f"\nTest type distribution:")
    for test_type, count in massive_df['test_type'].value_counts().head(10).items():
        percentage = count / len(massive_df) * 100
        print(f"  {test_type}: {count:,} ({percentage:.1f}%)")
    
    print(f"\nDifficulty distribution:")
    for difficulty, count in massive_df['difficulty'].value_counts().items():
        percentage = count / len(massive_df) * 100
        print(f"  {difficulty}: {count:,} ({percentage:.1f}%)")
    
    print(f"\n💾 Dataset saved to: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

if __name__ == "__main__":
    main()