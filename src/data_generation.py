import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import pandas as pd
import numpy as np
import random
from typing import List, Dict, Tuple
import logging
from pathlib import Path
import json
import re
from collections import defaultdict
from tqdm import tqdm

logger = logging.getLogger(__name__)

class LargeDatasetGenerator:
    """
    Generate large-scale training dataset for cross-encoder model.
    Creates positive and negative pairs with various difficulty levels.
    """
    
    def __init__(self, corpus_path: str):
        self.corpus_df = pd.read_csv(corpus_path)
        self.passages_by_work = defaultdict(list)
        self.passages_by_category = defaultdict(list)
        self._index_passages()
    
    def _index_passages(self):
        """Index passages by work and category for efficient sampling."""
        for idx, row in self.corpus_df.iterrows():
            work = row.get('work', 'unknown')
            category = row.get('category', 'unknown')
            
            passage_data = {
                'idx': idx,
                'text': str(row['text']),
                'work': work,
                'category': category,
                'length': len(str(row['text']))
            }
            
            self.passages_by_work[work].append(passage_data)
            self.passages_by_category[category].append(passage_data)
    
    def create_query_variants(self, text: str) -> List[str]:
        """Create multiple query variants from a passage."""
        words = text.split()
        variants = []
        
        # Full text
        variants.append(text)
        
        # Different lengths
        if len(words) >= 4:
            # First quarter
            variants.append(' '.join(words[:len(words)//4]))
            # First half
            variants.append(' '.join(words[:len(words)//2]))
            # First three quarters
            variants.append(' '.join(words[:3*len(words)//4]))
            # Last half
            variants.append(' '.join(words[len(words)//2:]))
            # Middle portion
            start = len(words)//4
            end = 3*len(words)//4
            variants.append(' '.join(words[start:end]))
        
        # Sliding windows
        if len(words) >= 6:
            window_size = min(5, len(words)//2)
            for i in range(0, len(words) - window_size + 1, 2):
                variants.append(' '.join(words[i:i+window_size]))
        
        # Random subsets
        if len(words) >= 8:
            for _ in range(3):
                subset_size = random.randint(3, min(7, len(words)-1))
                start = random.randint(0, len(words) - subset_size)
                variants.append(' '.join(words[start:start+subset_size]))
        
        return list(set(variants))  # Remove duplicates
    
    def add_noise_to_query(self, query: str, noise_level: float = 0.1) -> str:
        """Add noise to query to simulate OCR errors or variations."""
        words = query.split()
        noisy_words = []
        
        for word in words:
            if random.random() < noise_level and len(word) > 3:
                # Character substitution
                if random.random() < 0.5:
                    pos = random.randint(0, len(word)-1)
                    chars = list(word)
                    # Common OCR substitutions
                    substitutions = {
                        'a': 'ā', 'i': 'ī', 'u': 'ū', 'r': 'ṛ',
                        'n': 'ṇ', 's': 'ś', 't': 'ṭ', 'd': 'ḍ'
                    }
                    if chars[pos] in substitutions:
                        chars[pos] = substitutions[chars[pos]]
                    word = ''.join(chars)
                
                # Character deletion
                elif random.random() < 0.3 and len(word) > 4:
                    pos = random.randint(1, len(word)-2)
                    word = word[:pos] + word[pos+1:]
            
            noisy_words.append(word)
        
        return ' '.join(noisy_words)
    
    def generate_positive_pairs(self, num_pairs: int = 10000) -> List[Dict]:
        """Generate positive query-passage pairs."""
        positive_pairs = []
        
        # Sample passages
        sampled_passages = self.corpus_df.sample(n=min(num_pairs//5, len(self.corpus_df)))
        
        for _, passage_row in tqdm(sampled_passages.iterrows(), total=len(sampled_passages), desc="Generating positive pairs"):
            text = str(passage_row['text'])
            
            # Skip very short texts
            if len(text.split()) < 4:
                continue
            
            # Create query variants
            query_variants = self.create_query_variants(text)
            
            for query in query_variants:
                if len(query.split()) >= 2:  # Ensure minimum query length
                    pair = {
                        'query': query,
                        'passage': text,
                        'label': 1,
                        'pair_type': 'positive',
                        'work': passage_row.get('work', 'unknown'),
                        'category': passage_row.get('category', 'unknown'),
                        'difficulty': self._calculate_difficulty(query, text)
                    }
                    positive_pairs.append(pair)
                
                # Add noisy versions
                if random.random() < 0.3:
                    noisy_query = self.add_noise_to_query(query)
                    if noisy_query != query:
                        pair = {
                            'query': noisy_query,
                            'passage': text,
                            'label': 1,
                            'pair_type': 'positive_noisy',
                            'work': passage_row.get('work', 'unknown'),
                            'category': passage_row.get('category', 'unknown'),
                            'difficulty': self._calculate_difficulty(noisy_query, text)
                        }
                        positive_pairs.append(pair)
                
                if len(positive_pairs) >= num_pairs:
                    break
            
            if len(positive_pairs) >= num_pairs:
                break
        
        return positive_pairs[:num_pairs]
    
    def generate_hard_negatives(self, positive_pairs: List[Dict], 
                               negatives_per_positive: int = 3) -> List[Dict]:
        """Generate hard negative pairs."""
        negative_pairs = []
        
        for pos_pair in tqdm(positive_pairs, desc="Generating hard negatives"):
            query = pos_pair['query']
            pos_work = pos_pair['work']
            pos_category = pos_pair['category']
            
            # Generate different types of hard negatives
            hard_negatives = []
            
            # 1. Same work, different passage
            if pos_work in self.passages_by_work and len(self.passages_by_work[pos_work]) > 1:
                candidates = [p for p in self.passages_by_work[pos_work] 
                            if p['text'] != pos_pair['passage']]
                if candidates:
                    neg_passage = random.choice(candidates)
                    hard_negatives.append({
                        'passage': neg_passage['text'],
                        'type': 'same_work',
                        'work': neg_passage['work'],
                        'category': neg_passage['category']
                    })
            
            # 2. Same category, different work
            if pos_category in self.passages_by_category:
                candidates = [p for p in self.passages_by_category[pos_category]
                            if p['work'] != pos_work]
                if candidates:
                    neg_passage = random.choice(candidates)
                    hard_negatives.append({
                        'passage': neg_passage['text'],
                        'type': 'same_category',
                        'work': neg_passage['work'],
                        'category': neg_passage['category']
                    })
            
            # 3. Similar length passages
            target_length = len(pos_pair['passage'])
            length_candidates = []
            for _, row in self.corpus_df.iterrows():
                text_len = len(str(row['text']))
                if (abs(text_len - target_length) < target_length * 0.3 and 
                    str(row['text']) != pos_pair['passage']):
                    length_candidates.append({
                        'text': str(row['text']),
                        'work': row.get('work', 'unknown'),
                        'category': row.get('category', 'unknown')
                    })
            
            if length_candidates:
                neg_passage = random.choice(length_candidates)
                hard_negatives.append({
                    'passage': neg_passage['text'],
                    'type': 'similar_length',
                    'work': neg_passage['work'],
                    'category': neg_passage['category']
                })
            
            # 4. Random negatives
            while len(hard_negatives) < negatives_per_positive:
                random_row = self.corpus_df.sample(1).iloc[0]
                if str(random_row['text']) != pos_pair['passage']:
                    hard_negatives.append({
                        'passage': str(random_row['text']),
                        'type': 'random',
                        'work': random_row.get('work', 'unknown'),
                        'category': random_row.get('category', 'unknown')
                    })
            
            # Create negative pairs
            for neg_data in hard_negatives[:negatives_per_positive]:
                pair = {
                    'query': query,
                    'passage': neg_data['passage'],
                    'label': 0,
                    'pair_type': f"negative_{neg_data['type']}",
                    'work': neg_data['work'],
                    'category': neg_data['category'],
                    'difficulty': self._calculate_difficulty(query, neg_data['passage'])
                }
                negative_pairs.append(pair)
        
        return negative_pairs
    
    def _calculate_difficulty(self, query: str, passage: str) -> str:
        """Calculate difficulty level of the pair."""
        query_words = set(query.lower().split())
        passage_words = set(passage.lower().split())
        
        if not query_words:
            return 'invalid'
        
        overlap = len(query_words.intersection(passage_words))
        overlap_ratio = overlap / len(query_words)
        
        if overlap_ratio >= 0.8:
            return 'easy'
        elif overlap_ratio >= 0.5:
            return 'medium'
        elif overlap_ratio >= 0.2:
            return 'hard'
        else:
            return 'very_hard'
    
    def generate_large_dataset(self, 
                             num_positive_pairs: int = 50000,
                             negatives_per_positive: int = 3,
                             output_path: str = None) -> pd.DataFrame:
        """Generate complete large dataset."""
        logger.info(f"Generating {num_positive_pairs} positive pairs...")
        
        # Generate positive pairs
        positive_pairs = self.generate_positive_pairs(num_positive_pairs)
        logger.info(f"Generated {len(positive_pairs)} positive pairs")
        
        # Generate hard negatives
        logger.info(f"Generating hard negative pairs...")
        negative_pairs = self.generate_hard_negatives(positive_pairs, negatives_per_positive)
        logger.info(f"Generated {len(negative_pairs)} negative pairs")
        
        # Combine all pairs
        all_pairs = positive_pairs + negative_pairs
        random.shuffle(all_pairs)
        
        # Convert to DataFrame
        df = pd.DataFrame(all_pairs)
        
        # Add additional features
        df['query_length'] = df['query'].str.len()
        df['passage_length'] = df['passage'].str.len()
        df['length_ratio'] = df['query_length'] / df['passage_length']
        
        logger.info(f"Created dataset with {len(df)} total pairs")
        logger.info(f"Positive pairs: {len(df[df['label'] == 1])}")
        logger.info(f"Negative pairs: {len(df[df['label'] == 0])}")
        
        # Show difficulty distribution
        difficulty_dist = df['difficulty'].value_counts()
        logger.info(f"Difficulty distribution:\n{difficulty_dist}")
        
        # Save if path provided
        if output_path:
            df.to_csv(output_path, index=False)
            logger.info(f"Dataset saved to {output_path}")
        
        return df
    
    def create_balanced_dataset(self, 
                              total_pairs: int = 100000,
                              positive_ratio: float = 0.3,
                              difficulty_distribution: Dict[str, float] = None) -> pd.DataFrame:
        """Create a balanced dataset with controlled difficulty distribution."""
        
        if difficulty_distribution is None:
            difficulty_distribution = {
                'easy': 0.3,
                'medium': 0.4,
                'hard': 0.2,
                'very_hard': 0.1
            }
        
        num_positive = int(total_pairs * positive_ratio)
        num_negative = total_pairs - num_positive
        
        logger.info(f"Creating balanced dataset: {num_positive} pos, {num_negative} neg")
        
        # Generate positive pairs
        positive_pairs = self.generate_positive_pairs(num_positive * 2)  # Generate extra
        
        # Filter by difficulty
        filtered_positives = []
        for difficulty, target_ratio in tqdm(difficulty_distribution.items(), desc="Filtering by difficulty"):
            target_count = int(num_positive * target_ratio)
            candidates = [p for p in positive_pairs if p['difficulty'] == difficulty]
            selected = random.sample(candidates, min(target_count, len(candidates)))
            filtered_positives.extend(selected)
        
        # Ensure we have enough positives
        if len(filtered_positives) < num_positive:
            remaining = num_positive - len(filtered_positives)
            remaining_pairs = [p for p in positive_pairs if p not in filtered_positives]
            filtered_positives.extend(random.sample(remaining_pairs, 
                                                  min(remaining, len(remaining_pairs))))
        
        final_positives = filtered_positives[:num_positive]
        
        # Generate negatives
        negative_pairs = self.generate_hard_negatives(final_positives, 
                                                    negatives_per_positive=num_negative//num_positive + 1)
        final_negatives = random.sample(negative_pairs, min(num_negative, len(negative_pairs)))
        
        # Combine and shuffle
        all_pairs = final_positives + final_negatives
        random.shuffle(all_pairs)
        
        df = pd.DataFrame(all_pairs)
        
        logger.info(f"Balanced dataset created: {len(df)} pairs")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
        logger.info(f"Difficulty distribution:\n{df['difficulty'].value_counts()}")
        
        return df

def augment_existing_dataset(existing_path: str, corpus_path: str, 
                           target_size: int = 100000) -> pd.DataFrame:
    """Augment existing dataset to reach target size."""
    existing_df = pd.read_csv(existing_path)
    logger.info(f"Existing dataset size: {len(existing_df)}")
    
    if len(existing_df) >= target_size:
        logger.info("Dataset already large enough")
        return existing_df
    
    additional_needed = target_size - len(existing_df)
    
    generator = LargeDatasetGenerator(corpus_path)
    additional_df = generator.generate_large_dataset(additional_needed // 4)
    
    # Combine datasets
    combined_df = pd.concat([existing_df, additional_df], ignore_index=True)
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)  # Shuffle
    
    logger.info(f"Augmented dataset size: {len(combined_df)}")
    
    return combined_df

def main():
    """Generate large dataset for cross-encoder training."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate or augment cross-encoder training dataset')
    parser.add_argument('--augment', type=str, help='Path to existing dataset to augment')
    parser.add_argument('--target-size', type=int, default=100000, help='Target dataset size')
    parser.add_argument('--corpus', type=str, default="/home/sai/Desktop/ullu/data/sanskrit_quotes.csv", help='Corpus path')
    parser.add_argument('--output', type=str, default="/home/sai/Desktop/ullu/data/large_training_dataset.csv", help='Output path')
    
    args = parser.parse_args()
    
    logging.basicConfig(level=logging.INFO)
    
    if args.augment:
        # Augment existing dataset
        logger.info(f"Augmenting existing dataset: {args.augment}")
        large_df = augment_existing_dataset(args.augment, args.corpus, args.target_size)
        output_path = args.output
    else:
        # Generate new dataset
        corpus_path = args.corpus
        output_path = args.output
        
        # Initialize generator
        generator = LargeDatasetGenerator(corpus_path)
        
        # Generate large balanced dataset
        logger.info("Generating large training dataset...")
        
        large_df = generator.create_balanced_dataset(
            total_pairs=args.target_size,
            positive_ratio=0.4,
            difficulty_distribution={
                'easy': 0.2,
                'medium': 0.4,
                'hard': 0.3,
                'very_hard': 0.1
            }
        )
    
    # Save dataset
    large_df.to_csv(output_path, index=False)
    
    # Show statistics
    print("\n" + "="*60)
    print("LARGE DATASET GENERATION COMPLETE")
    print("="*60)
    print(f"Total pairs: {len(large_df):,}")
    print(f"Positive pairs: {len(large_df[large_df['label'] == 1]):,}")
    print(f"Negative pairs: {len(large_df[large_df['label'] == 0]):,}")
    
    print(f"\nDifficulty distribution:")
    for difficulty, count in large_df['difficulty'].value_counts().items():
        print(f"  {difficulty}: {count:,} ({count/len(large_df)*100:.1f}%)")
    
    print(f"\nPair type distribution:")
    for pair_type, count in large_df['pair_type'].value_counts().items():
        print(f"  {pair_type}: {count:,}")
    
    print(f"\nDataset saved to: {output_path}")
    
    # Show sample pairs
    print(f"\nSample positive pair:")
    pos_sample = large_df[large_df['label'] == 1].iloc[0]
    print(f"Query: {pos_sample['query']}")
    print(f"Passage: {pos_sample['passage'][:100]}...")
    print(f"Difficulty: {pos_sample['difficulty']}")
    
    print(f"\nSample negative pair:")
    neg_sample = large_df[large_df['label'] == 0].iloc[0]
    print(f"Query: {neg_sample['query']}")
    print(f"Passage: {neg_sample['passage'][:100]}...")
    print(f"Difficulty: {neg_sample['difficulty']}")

if __name__ == "__main__":
    main()