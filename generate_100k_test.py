#!/usr/bin/env python3
"""
Generate 100K Test Dataset for Cross-Encoder Evaluation
Optimized for overnight batch processing with progress tracking.
"""

import pandas as pd
import numpy as np
import random
import logging
import time
import os
from pathlib import Path
from typing import List, Dict
import gc  # Garbage collection for memory management

# Setup logging for overnight run
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data/test_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Optimized100KTestGenerator:
    """Optimized generator for 100K test dataset with memory management."""
    
    def __init__(self):
        self.batch_size = 1000  # Process in batches to manage memory
        self.checkpoint_interval = 10000  # Save checkpoints every 10K
        self.all_test_cases = []
        
    def load_corpus_efficiently(self):
        """Load corpus with memory optimization."""
        logger.info("Loading corpus data...")
        
        # Load quotes dataset
        quotes_path = "data/sanskrit_quotes.csv"
        self.quotes_df = pd.read_csv(quotes_path)
        logger.info(f"Loaded quotes: {len(self.quotes_df):,}")
        
        # Load large corpus if available
        corpus_path = "data/sanskrit_corpus.csv"
        if Path(corpus_path).exists():
            logger.info("Loading large corpus in chunks...")
            # Read in chunks to manage memory
            chunk_size = 50000
            corpus_chunks = []
            
            for chunk in pd.read_csv(corpus_path, chunksize=chunk_size, dtype={'text': str}):
                # Clean and filter chunk
                chunk = chunk.dropna(subset=['text'])
                chunk = chunk[chunk['text'].str.len() > 20]  # Filter very short texts
                corpus_chunks.append(chunk)
                
                if len(corpus_chunks) * chunk_size >= 200000:  # Limit to 200K for memory
                    break
            
            self.corpus_df = pd.concat(corpus_chunks, ignore_index=True)
            logger.info(f"Loaded large corpus: {len(self.corpus_df):,}")
            
            # Clear chunks from memory
            del corpus_chunks
            gc.collect()
        else:
            self.corpus_df = self.quotes_df
            logger.info("Using quotes corpus as main corpus")
        
        # Index for efficient sampling
        self._create_indices()
    
    def _create_indices(self):
        """Create indices for efficient sampling."""
        logger.info("Creating indices for efficient sampling...")
        
        self.work_index = {}
        self.category_index = {}
        self.length_index = {}
        
        for idx, row in self.corpus_df.iterrows():
            work = str(row.get('work', 'unknown'))
            category = str(row.get('category', 'unknown'))
            text_length = len(str(row.get('text', '')))
            length_bucket = (text_length // 100) * 100  # 100-char buckets
            
            # Work index
            if work not in self.work_index:
                self.work_index[work] = []
            self.work_index[work].append(idx)
            
            # Category index
            if category not in self.category_index:
                self.category_index[category] = []
            self.category_index[category].append(idx)
            
            # Length index
            if length_bucket not in self.length_index:
                self.length_index[length_bucket] = []
            self.length_index[length_bucket].append(idx)
        
        logger.info(f"Indexed {len(self.work_index)} works, {len(self.category_index)} categories")
    
    def generate_exact_matches_batch(self, target_count: int, progress_callback=None) -> List[Dict]:
        """Generate exact match test cases in batches."""
        logger.info(f"Generating {target_count:,} exact match cases...")
        
        test_cases = []
        processed = 0
        
        # Sample passages for exact matches
        sample_size = min(target_count // 4, len(self.corpus_df))  # 4 queries per passage
        sampled_passages = self.corpus_df.sample(n=sample_size, random_state=42)
        
        for idx, row in sampled_passages.iterrows():
            if len(test_cases) >= target_count:
                break
                
            text = str(row.get('text', ''))
            words = text.split()
            
            if len(words) < 4:
                continue
            
            # Generate multiple exact match queries per passage
            for ratio in [0.2, 0.35, 0.5, 0.65]:
                query_len = max(2, int(len(words) * ratio))
                
                # Multiple starting positions
                for start_factor in [0.0, 0.25, 0.5]:
                    start_pos = int((len(words) - query_len) * start_factor)
                    if start_pos + query_len <= len(words):
                        query = ' '.join(words[start_pos:start_pos + query_len])
                        
                        if len(query.strip()) > 15:  # Minimum query length
                            test_cases.append({
                                'query': query,
                                'passage': text,
                                'label': 1,
                                'test_type': 'exact_match',
                                'expected_score': 'very_high',
                                'difficulty': 'easy',
                                'work': str(row.get('work', 'unknown')),
                                'category': str(row.get('category', 'unknown')),
                                'query_ratio': ratio,
                                'start_position': start_factor
                            })
                            
                            if len(test_cases) >= target_count:
                                break
                
                if len(test_cases) >= target_count:
                    break
            
            processed += 1
            if processed % 1000 == 0 and progress_callback:
                progress_callback(f"Exact matches: {len(test_cases):,}/{target_count:,}")
        
        logger.info(f"Generated {len(test_cases):,} exact match cases")
        return test_cases[:target_count]
    
    def generate_partial_matches_batch(self, target_count: int, progress_callback=None) -> List[Dict]:
        """Generate partial match test cases in batches."""
        logger.info(f"Generating {target_count:,} partial match cases...")
        
        test_cases = []
        processed = 0
        
        # Sample passages for partial matches
        sample_size = min(target_count // 6, len(self.corpus_df))  # 6 queries per passage
        sampled_passages = self.corpus_df.sample(n=sample_size, random_state=43)
        
        for idx, row in sampled_passages.iterrows():
            if len(test_cases) >= target_count:
                break
                
            text = str(row.get('text', ''))
            words = text.split()
            
            if len(words) < 8:
                continue
            
            # Generate multiple partial match variants
            for overlap_ratio in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                query_length = random.randint(5, 10)
                num_overlapping = max(2, int(query_length * overlap_ratio))
                
                # Sample overlapping words from passage
                overlapping_words = random.sample(words, min(num_overlapping, len(words)))
                
                # Add non-overlapping words from other passages
                remaining_slots = query_length - len(overlapping_words)
                if remaining_slots > 0:
                    # Sample from multiple other passages for diversity
                    other_words = []
                    for _ in range(min(3, remaining_slots)):
                        other_passage = self.corpus_df.sample(1).iloc[0]
                        other_text_words = str(other_passage.get('text', '')).split()
                        if other_text_words:
                            word = random.choice(other_text_words)
                            if word not in overlapping_words:
                                other_words.append(word)
                    
                    overlapping_words.extend(other_words)
                
                # Create query
                random.shuffle(overlapping_words)
                query = ' '.join(overlapping_words[:query_length])
                
                if len(query.strip()) > 15:
                    difficulty = 'medium' if overlap_ratio >= 0.5 else 'hard'
                    expected_score = 'medium-high' if overlap_ratio >= 0.6 else 'medium'
                    
                    test_cases.append({
                        'query': query,
                        'passage': text,
                        'label': 1,
                        'test_type': 'partial_match',
                        'expected_score': expected_score,
                        'difficulty': difficulty,
                        'work': str(row.get('work', 'unknown')),
                        'category': str(row.get('category', 'unknown')),
                        'overlap_ratio': overlap_ratio,
                        'query_length': query_length
                    })
                    
                    if len(test_cases) >= target_count:
                        break
            
            processed += 1
            if processed % 500 == 0 and progress_callback:
                progress_callback(f"Partial matches: {len(test_cases):,}/{target_count:,}")
        
        logger.info(f"Generated {len(test_cases):,} partial match cases")
        return test_cases[:target_count]
    
    def generate_semantic_matches_batch(self, target_count: int, progress_callback=None) -> List[Dict]:
        """Generate semantic match test cases."""
        logger.info(f"Generating {target_count:,} semantic match cases...")
        
        test_cases = []
        
        # Extended Sanskrit semantic groups
        semantic_groups = {
            'viṣṇu': ['hari', 'nārāyaṇa', 'keśava', 'madhusūdana', 'govinda', 'kṛṣṇa', 'vāsudeva', 'janārdana'],
            'śiva': ['maheśvara', 'śaṅkara', 'rudra', 'īśa', 'hara', 'śambhu', 'tryambaka', 'natarāja'],
            'brahmā': ['pitāmaha', 'svayambhū', 'prajāpati', 'virañci', 'caturmukha', 'kamalāsana'],
            'devī': ['ambā', 'umā', 'pārvatī', 'śakti', 'bhagavatī', 'mātā', 'durgā', 'lakṣmī'],
            'dharma': ['nyāya', 'satya', 'rita', 'yuga', 'svadharma', 'rājadharma'],
            'karma': ['kriyā', 'cikīrṣā', 'pravṛtti', 'ceṣṭā', 'karmaphala'],
            'mokṣa': ['mukti', 'nirvāṇa', 'kaivalya', 'apavarga', 'videhamukti'],
            'yoga': ['yuj', 'samādhi', 'dhyāna', 'ekāgratā', 'yogasādhana'],
            'rāja': ['nṛpa', 'bhūpati', 'bhūpāla', 'mahīpati', 'kṣatriya', 'narendrā'],
            'guru': ['ācārya', 'śikṣaka', 'upādhyāya', 'preceptor'],
            'mantra': ['gāyatrī', 'ṛk', 'yajus', 'sāman', 'japa'],
            'yajña': ['homa', 'havana', 'agnihotra', 'sacrifice']
        }
        
        for primary_term, related_terms in semantic_groups.items():
            if len(test_cases) >= target_count:
                break
                
            # Find passages containing primary term
            matching_passages = self.corpus_df[
                self.corpus_df['text'].str.contains(primary_term, case=False, na=False)
            ]
            
            if len(matching_passages) == 0:
                continue
            
            # Sample passages for this term
            sample_size = min(len(matching_passages), target_count // len(semantic_groups))
            sampled = matching_passages.sample(n=sample_size, random_state=44)
            
            for _, row in sampled.iterrows():
                if len(test_cases) >= target_count:
                    break
                    
                text = str(row.get('text', ''))
                
                for related_term in related_terms[:4]:  # Use first 4 related terms
                    # Method 1: Direct substitution
                    words = text.lower().split()
                    query_words = []
                    
                    for word in words[:8]:  # Take first 8 words
                        if primary_term in word:
                            query_words.append(word.replace(primary_term, related_term))
                        else:
                            query_words.append(word)
                    
                    query = ' '.join(query_words)
                    
                    if len(query.strip()) > 15:
                        test_cases.append({
                            'query': query,
                            'passage': text,
                            'label': 1,
                            'test_type': 'semantic_match',
                            'expected_score': 'medium-high',
                            'difficulty': 'hard',
                            'work': str(row.get('work', 'unknown')),
                            'category': str(row.get('category', 'unknown')),
                            'primary_term': primary_term,
                            'related_term': related_term,
                            'method': 'substitution'
                        })
                        
                        if len(test_cases) >= target_count:
                            break
                    
                    # Method 2: Context query
                    context_query = f"namaḥ {related_term}"
                    test_cases.append({
                        'query': context_query,
                        'passage': text,
                        'label': 1,
                        'test_type': 'semantic_context',
                        'expected_score': 'medium',
                        'difficulty': 'hard',
                        'work': str(row.get('work', 'unknown')),
                        'category': str(row.get('category', 'unknown')),
                        'primary_term': primary_term,
                        'related_term': related_term,
                        'method': 'context'
                    })
                    
                    if len(test_cases) >= target_count:
                        break
                
                if len(test_cases) >= target_count:
                    break
            
            if progress_callback:
                progress_callback(f"Semantic matches: {len(test_cases):,}/{target_count:,}")
        
        logger.info(f"Generated {len(test_cases):,} semantic match cases")
        return test_cases[:target_count]
    
    def generate_negative_cases_batch(self, positive_cases: List[Dict], 
                                    target_count: int, progress_callback=None) -> List[Dict]:
        """Generate negative test cases efficiently."""
        logger.info(f"Generating {target_count:,} negative cases...")
        
        negative_cases = []
        
        # Sample positive cases to create negatives from
        sampled_positives = random.sample(positive_cases, min(target_count // 4, len(positive_cases)))
        
        negative_types = [
            ('random', 0.4),           # 40% random
            ('same_category', 0.3),    # 30% same category  
            ('same_work', 0.2),        # 20% same work
            ('similar_length', 0.1)    # 10% similar length
        ]
        
        for pos_case in sampled_positives:
            if len(negative_cases) >= target_count:
                break
                
            query = pos_case['query']
            pos_work = pos_case['work']
            pos_category = pos_case['category']
            pos_length = len(pos_case['passage'])
            
            for neg_type, ratio in negative_types:
                num_for_type = max(1, int(4 * ratio))  # 4 negatives per positive
                
                for _ in range(num_for_type):
                    if len(negative_cases) >= target_count:
                        break
                        
                    try:
                        if neg_type == 'random':
                            neg_passage = self.corpus_df.sample(1).iloc[0]
                            difficulty = 'easy'
                            
                        elif neg_type == 'same_category':
                            if pos_category in self.category_index:
                                candidates = self.category_index[pos_category]
                                candidates = [c for c in candidates if self.corpus_df.loc[c, 'work'] != pos_work]
                                if candidates:
                                    neg_idx = random.choice(candidates)
                                    neg_passage = self.corpus_df.loc[neg_idx]
                                    difficulty = 'medium'
                                else:
                                    continue
                            else:
                                continue
                                
                        elif neg_type == 'same_work':
                            if pos_work in self.work_index:
                                candidates = self.work_index[pos_work]
                                candidates = [c for c in candidates if str(self.corpus_df.loc[c, 'text']) != pos_case['passage']]
                                if candidates:
                                    neg_idx = random.choice(candidates)
                                    neg_passage = self.corpus_df.loc[neg_idx]
                                    difficulty = 'hard'
                                else:
                                    continue
                            else:
                                continue
                                
                        elif neg_type == 'similar_length':
                            length_bucket = (pos_length // 100) * 100
                            candidates = []
                            
                            # Check nearby length buckets
                            for offset in [0, 100, -100, 200, -200]:
                                bucket = length_bucket + offset
                                if bucket in self.length_index:
                                    candidates.extend(self.length_index[bucket])
                            
                            if candidates:
                                neg_idx = random.choice(candidates)
                                neg_passage = self.corpus_df.loc[neg_idx]
                                difficulty = 'medium'
                            else:
                                continue
                        
                        # Ensure different passage
                        if str(neg_passage.get('text', '')) != pos_case['passage']:
                            negative_cases.append({
                                'query': query,
                                'passage': str(neg_passage.get('text', '')),
                                'label': 0,
                                'test_type': f'negative_{neg_type}',
                                'expected_score': 'very_low',
                                'difficulty': difficulty,
                                'work': str(neg_passage.get('work', 'unknown')),
                                'category': str(neg_passage.get('category', 'unknown')),
                                'negative_type': neg_type
                            })
                        
                    except Exception as e:
                        logger.warning(f"Error generating {neg_type} negative: {e}")
                        continue
            
            if len(negative_cases) % 5000 == 0 and progress_callback:
                progress_callback(f"Negative cases: {len(negative_cases):,}/{target_count:,}")
        
        logger.info(f"Generated {len(negative_cases):,} negative cases")
        return negative_cases[:target_count]
    
    def save_checkpoint(self, test_cases: List[Dict], checkpoint_num: int):
        """Save checkpoint during generation."""
        checkpoint_path = f"data/checkpoint_{checkpoint_num}.csv"
        df = pd.DataFrame(test_cases)
        df.to_csv(checkpoint_path, index=False)
        logger.info(f"Checkpoint {checkpoint_num} saved: {len(test_cases):,} cases")
    
    def generate_100k_dataset(self) -> pd.DataFrame:
        """Generate complete 100K test dataset with progress tracking."""
        start_time = time.time()
        logger.info("Starting 100K test dataset generation...")
        
        # Load data
        self.load_corpus_efficiently()
        
        # Define target sizes (total = 100K)
        targets = {
            'exact_match': 30000,      # 30%
            'partial_match': 40000,    # 40% 
            'semantic_match': 10000,   # 10%
            'negative': 20000          # 20%
        }
        
        def progress_callback(message):
            elapsed = time.time() - start_time
            logger.info(f"[{elapsed/3600:.1f}h] {message}")
        
        all_test_cases = []
        
        # Generate each type with progress tracking
        logger.info("Phase 1/4: Exact matches...")
        exact_cases = self.generate_exact_matches_batch(targets['exact_match'], progress_callback)
        all_test_cases.extend(exact_cases)
        self.save_checkpoint(all_test_cases, 1)
        
        logger.info("Phase 2/4: Partial matches...")
        partial_cases = self.generate_partial_matches_batch(targets['partial_match'], progress_callback)
        all_test_cases.extend(partial_cases)
        self.save_checkpoint(all_test_cases, 2)
        
        logger.info("Phase 3/4: Semantic matches...")
        semantic_cases = self.generate_semantic_matches_batch(targets['semantic_match'], progress_callback)
        all_test_cases.extend(semantic_cases)
        self.save_checkpoint(all_test_cases, 3)
        
        logger.info("Phase 4/4: Negative cases...")
        positive_cases = exact_cases + partial_cases + semantic_cases
        negative_cases = self.generate_negative_cases_batch(positive_cases, targets['negative'], progress_callback)
        all_test_cases.extend(negative_cases)
        
        # Final shuffle and DataFrame creation
        logger.info("Creating final dataset...")
        random.shuffle(all_test_cases)
        
        # Ensure exactly 100K
        all_test_cases = all_test_cases[:100000]
        
        test_df = pd.DataFrame(all_test_cases)
        test_df['test_id'] = range(len(test_df))
        test_df['dataset_version'] = '100k_v1'
        test_df['generation_time'] = time.time()
        
        total_time = time.time() - start_time
        logger.info(f"Dataset generation completed in {total_time/3600:.2f} hours")
        
        return test_df

def main():
    """Generate 100K test dataset for overnight run."""
    print("🌙 OVERNIGHT 100K Test Dataset Generation")
    print("This will generate 100,000 high-quality test pairs")
    print("Estimated time: 2-6 hours depending on system")
    print("=" * 60)
    
    # Create data directory if needed
    Path("data").mkdir(exist_ok=True)
    
    # Initialize generator
    generator = Optimized100KTestGenerator()
    
    try:
        # Generate dataset
        test_df = generator.generate_100k_dataset()
        
        # Save final dataset
        output_path = "data/100k_cross_encoder_test.csv"
        test_df.to_csv(output_path, index=False)
        
        # Generate summary
        print(f"\n🎉 100K DATASET GENERATION COMPLETE!")
        print("=" * 50)
        print(f"Total pairs: {len(test_df):,}")
        print(f"Positive pairs: {len(test_df[test_df['label'] == 1]):,}")
        print(f"Negative pairs: {len(test_df[test_df['label'] == 0]):,}")
        
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
        
        logger.info("100K dataset generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during generation: {e}")
        print(f"❌ Generation failed: {e}")
        print("Check data/test_generation.log for details")

if __name__ == "__main__":
    main()