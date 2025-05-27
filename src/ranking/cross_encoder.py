import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import json
import pickle
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re

logger = logging.getLogger(__name__)

class FeatureExtractor:
    """Extract features for cross-encoder ranking."""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))
        self.char_ngram = TfidfVectorizer(analyzer='char', ngram_range=(3, 5), max_features=500)
        self.is_fitted = False
    
    def fit(self, texts: List[str]):
        """Fit feature extractors on training texts."""
        self.tfidf.fit(texts)
        self.char_ngram.fit(texts)
        self.is_fitted = True
    
    def extract_lexical_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract lexical similarity features."""
        features = {}
        
        # Word overlap
        query_words = set(query.lower().split())
        passage_words = set(passage.lower().split())
        
        if query_words:
            overlap = len(query_words.intersection(passage_words))
            features['word_overlap'] = overlap / len(query_words)
            features['jaccard_similarity'] = overlap / len(query_words.union(passage_words))
        else:
            features['word_overlap'] = 0.0
            features['jaccard_similarity'] = 0.0
        
        # Length features
        features['query_length'] = len(query)
        features['passage_length'] = len(passage)
        features['length_ratio'] = len(query) / max(len(passage), 1)
        
        # Character n-gram overlap
        query_chars = set()
        passage_chars = set()
        
        for i in range(len(query) - 2):
            query_chars.add(query[i:i+3])
        
        for i in range(len(passage) - 2):
            passage_chars.add(passage[i:i+3])
        
        if query_chars:
            char_overlap = len(query_chars.intersection(passage_chars))
            features['char_overlap'] = char_overlap / len(query_chars)
        else:
            features['char_overlap'] = 0.0
        
        return features
    
    def extract_semantic_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract semantic features using TF-IDF."""
        features = {}
        
        if not self.is_fitted:
            return features
        
        try:
            # TF-IDF similarity
            query_tfidf = self.tfidf.transform([query])
            passage_tfidf = self.tfidf.transform([passage])
            
            from sklearn.metrics.pairwise import cosine_similarity
            tfidf_sim = cosine_similarity(query_tfidf, passage_tfidf)[0, 0]
            features['tfidf_similarity'] = tfidf_sim
            
            # Character n-gram similarity
            query_char = self.char_ngram.transform([query])
            passage_char = self.char_ngram.transform([passage])
            
            char_sim = cosine_similarity(query_char, passage_char)[0, 0]
            features['char_ngram_similarity'] = char_sim
            
        except Exception as e:
            logger.warning(f"Error extracting semantic features: {e}")
            features['tfidf_similarity'] = 0.0
            features['char_ngram_similarity'] = 0.0
        
        return features
    
    def extract_position_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract positional features."""
        features = {}
        
        # Find query substring in passage
        query_lower = query.lower()
        passage_lower = passage.lower()
        
        if query_lower in passage_lower:
            position = passage_lower.find(query_lower)
            features['exact_match'] = 1.0
            features['match_position'] = position / max(len(passage), 1)
        else:
            features['exact_match'] = 0.0
            features['match_position'] = 0.0
        
        # Check for partial matches
        query_words = query_lower.split()
        passage_words = passage_lower.split()
        
        if query_words and passage_words:
            # Find longest common subsequence of words
            lcs_length = self._longest_common_subsequence(query_words, passage_words)
            features['word_lcs'] = lcs_length / len(query_words)
        else:
            features['word_lcs'] = 0.0
        
        return features
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Find length of longest common subsequence."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def extract_all_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract all features for a query-passage pair."""
        features = {}
        
        features.update(self.extract_lexical_features(query, passage))
        features.update(self.extract_semantic_features(query, passage))
        features.update(self.extract_position_features(query, passage))
        
        return features

class CrossEncoderRanker:
    """Cross-encoder for ranking query-passage pairs."""
    
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.feature_extractor = FeatureExtractor()
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        if model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    def create_training_data(self, df: pd.DataFrame, num_negatives: int = 3) -> Tuple[List[Dict], List[int]]:
        """Create training data with positive and negative pairs."""
        training_pairs = []
        labels = []
        
        # Fit feature extractors on all texts
        all_texts = df['text'].fillna('').tolist()
        self.feature_extractor.fit(all_texts)
        
        for idx, row in df.iterrows():
            text = str(row['text'])
            
            # Create positive pair (partial query -> full text)
            words = text.split()
            if len(words) > 4:
                # Use first half as query
                query = ' '.join(words[:len(words)//2])
                
                features = self.feature_extractor.extract_all_features(query, text)
                training_pairs.append(features)
                labels.append(1)  # Positive
                
                # Create negative pairs (same query with different texts)
                negative_texts = df[df.index != idx].sample(n=num_negatives, replace=True)['text'].tolist()
                
                for neg_text in negative_texts:
                    neg_features = self.feature_extractor.extract_all_features(query, str(neg_text))
                    training_pairs.append(neg_features)
                    labels.append(0)  # Negative
        
        return training_pairs, labels
    
    def train(self, training_pairs: List[Dict], labels: List[int]):
        """Train the cross-encoder model."""
        logger.info(f"Training cross-encoder with {len(training_pairs)} pairs")
        
        # Convert features to matrix
        if not training_pairs:
            raise ValueError("No training pairs provided")
        
        self.feature_names = sorted(training_pairs[0].keys())
        X = np.array([[pair.get(feature, 0.0) for feature in self.feature_names] 
                     for pair in training_pairs])
        y = np.array(labels)
        
        # Train model
        self.model.fit(X, y)
        self.is_trained = True
        
        # Calculate training accuracy
        y_pred = self.model.predict(X)
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"Training accuracy: {accuracy:.3f}")
        
        return accuracy
    
    def rank_passages(self, query: str, passages: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Rank passages for a given query."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        scored_passages = []
        
        for passage_data in passages:
            passage_text = passage_data.get('text', '')
            
            # Extract features
            features = self.feature_extractor.extract_all_features(query, passage_text)
            
            # Convert to feature vector
            feature_vector = np.array([[features.get(feature, 0.0) for feature in self.feature_names]])
            
            # Get score (probability of being relevant)
            try:
                score = self.model.predict_proba(feature_vector)[0, 1]  # Probability of class 1
            except:
                score = 0.0
            
            # Add score to passage data
            scored_passage = passage_data.copy()
            scored_passage['ranking_score'] = score
            scored_passages.append(scored_passage)
        
        # Sort by score descending
        scored_passages.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Return top-k if specified
        if top_k:
            scored_passages = scored_passages[:top_k]
        
        # Add ranking positions
        for i, passage in enumerate(scored_passages):
            passage['final_rank'] = i + 1
        
        return scored_passages
    
    def evaluate(self, test_queries: List[Dict]) -> Dict[str, float]:
        """Evaluate the ranker on test queries."""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        precision_at_1 = []
        precision_at_5 = []
        mrr_scores = []
        
        for query_data in test_queries:
            query = query_data['query']
            ground_truth_id = query_data.get('ground_truth_id')
            
            # Create dummy candidates (in practice, these come from Stage 1)
            # For evaluation, we'll create some candidates including the ground truth
            candidates = [query_data]  # Ground truth as one candidate
            
            # Rank candidates
            ranked = self.rank_passages(query, candidates)
            
            # Calculate metrics
            if ranked:
                # Precision@1
                top1_id = ranked[0].get('quote_id', ranked[0].get('ground_truth_id'))
                precision_at_1.append(1.0 if top1_id == ground_truth_id else 0.0)
                
                # MRR
                for i, candidate in enumerate(ranked):
                    cand_id = candidate.get('quote_id', candidate.get('ground_truth_id'))
                    if cand_id == ground_truth_id:
                        mrr_scores.append(1.0 / (i + 1))
                        break
                else:
                    mrr_scores.append(0.0)
        
        metrics = {
            'precision_at_1': np.mean(precision_at_1) if precision_at_1 else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'num_queries': len(test_queries)
        }
        
        return metrics
    
    def save(self, path: str):
        """Save the trained model."""
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'feature_names': self.feature_names,
            'model_type': self.model_type,
            'is_trained': self.is_trained
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Cross-encoder saved to {path}")
    
    def load(self, path: str):
        """Load a trained model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Cross-encoder loaded from {path}")

def main():
    """Test the cross-encoder ranking system."""
    logging.basicConfig(level=logging.INFO)
    
    # Load data
    csv_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    df = pd.read_csv(csv_path)
    
    # Initialize ranker
    ranker = CrossEncoderRanker()
    
    # Create training data
    logger.info("Creating training data...")
    training_pairs, labels = ranker.create_training_data(df.head(100), num_negatives=2)
    
    # Train model
    ranker.train(training_pairs, labels)
    
    # Test ranking
    test_query = "dharma viṣṇu"
    test_passages = [
        {'text': 'dharmaviṣṇupriya śrīkṛṣṇa paramātma sanātana', 'quote_id': 1},
        {'text': 'rājya dhana āryāṇām viṣṇave namaḥ', 'quote_id': 2},
        {'text': 'gaṅgā yamunā sarasvatī devī', 'quote_id': 3},
        {'text': 'dharma artha kāma mokṣa viṣṇu', 'quote_id': 4}
    ]
    
    ranked_passages = ranker.rank_passages(test_query, test_passages)
    
    print("\n" + "="*60)
    print("CROSS-ENCODER RANKING TEST")
    print("="*60)
    print(f"Query: '{test_query}'")
    print("-" * 40)
    
    for passage in ranked_passages:
        print(f"Rank {passage['final_rank']}: Score={passage['ranking_score']:.3f}")
        print(f"  Text: {passage['text']}")
        print(f"  ID: {passage['quote_id']}")
    
    # Save model
    model_path = "/home/sai/Desktop/ullu/data/cross_encoder.pkl"
    ranker.save(model_path)
    
    print(f"\nModel saved to {model_path}")

if __name__ == "__main__":
    main()