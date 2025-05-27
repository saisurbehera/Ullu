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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import re

# Try to import XGBoost, fallback to LightGBM, then Random Forest
try:
    import xgboost as xgb
    USE_XGBOOST = True
    USE_LIGHTGBM = False
except ImportError:
    try:
        import lightgbm as lgb
        USE_XGBOOST = False
        USE_LIGHTGBM = True
    except ImportError:
        from sklearn.ensemble import RandomForestClassifier
        USE_XGBOOST = False
        USE_LIGHTGBM = False

logger = logging.getLogger(__name__)

class AdvancedFeatureExtractor:
    """Enhanced feature extractor for cross-encoder ranking."""
    
    def __init__(self):
        self.tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1, 3))
        self.char_ngram = TfidfVectorizer(analyzer='char', ngram_range=(3, 6), max_features=1000)
        self.word_ngram = TfidfVectorizer(analyzer='word', ngram_range=(2, 4), max_features=1000)
        self.is_fitted = False
        
        # Sanskrit-specific patterns
        self.sanskrit_patterns = {
            'deity_names': r'\b(viṣṇu|śiva|brahmā|kṛṣṇa|rāma|devī|gaṇeśa|hanumān)\b',
            'vedic_terms': r'\b(dharma|karma|mokṣa|yoga|mantra|yajña|soma)\b',
            'honorifics': r'\b(śrī|bhagavān|mahārāja|guru|ācārya)\b',
            'citations': r'//[^/]+//',
            'verse_markers': r'\|\||।।|॥'
        }
    
    def fit(self, texts: List[str]):
        """Fit feature extractors on training texts."""
        self.tfidf.fit(texts)
        self.char_ngram.fit(texts)
        self.word_ngram.fit(texts)
        self.is_fitted = True
    
    def extract_lexical_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract comprehensive lexical similarity features."""
        features = {}
        
        # Basic word overlap
        query_words = set(query.lower().split())
        passage_words = set(passage.lower().split())
        
        if query_words:
            overlap = len(query_words.intersection(passage_words))
            union = len(query_words.union(passage_words))
            
            features['word_overlap_ratio'] = overlap / len(query_words)
            features['jaccard_similarity'] = overlap / union if union > 0 else 0.0
            features['dice_coefficient'] = (2 * overlap) / (len(query_words) + len(passage_words))
            
            # Overlap for different word lengths
            short_query_words = {w for w in query_words if len(w) <= 4}
            long_query_words = {w for w in query_words if len(w) > 4}
            short_passage_words = {w for w in passage_words if len(w) <= 4}
            long_passage_words = {w for w in passage_words if len(w) > 4}
            
            if short_query_words:
                features['short_word_overlap'] = len(short_query_words.intersection(short_passage_words)) / len(short_query_words)
            else:
                features['short_word_overlap'] = 0.0
                
            if long_query_words:
                features['long_word_overlap'] = len(long_query_words.intersection(long_passage_words)) / len(long_query_words)
            else:
                features['long_word_overlap'] = 0.0
        else:
            features.update({
                'word_overlap_ratio': 0.0,
                'jaccard_similarity': 0.0,
                'dice_coefficient': 0.0,
                'short_word_overlap': 0.0,
                'long_word_overlap': 0.0
            })
        
        # Length features
        features['query_length'] = len(query)
        features['passage_length'] = len(passage)
        features['length_ratio'] = len(query) / max(len(passage), 1)
        features['length_diff'] = abs(len(query) - len(passage))
        features['query_word_count'] = len(query.split())
        features['passage_word_count'] = len(passage.split())
        features['word_count_ratio'] = len(query.split()) / max(len(passage.split()), 1)
        
        # Character-level features
        query_chars = set(query.lower())
        passage_chars = set(passage.lower())
        if query_chars:
            char_overlap = len(query_chars.intersection(passage_chars))
            features['char_overlap_ratio'] = char_overlap / len(query_chars)
        else:
            features['char_overlap_ratio'] = 0.0
        
        return features
    
    def extract_semantic_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract semantic features using multiple vectorizers."""
        features = {}
        
        if not self.is_fitted:
            return features
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            
            # TF-IDF similarity
            query_tfidf = self.tfidf.transform([query])
            passage_tfidf = self.tfidf.transform([passage])
            tfidf_sim = cosine_similarity(query_tfidf, passage_tfidf)[0, 0]
            features['tfidf_similarity'] = tfidf_sim
            
            # Character n-gram similarity
            query_char = self.char_ngram.transform([query])
            passage_char = self.char_ngram.transform([passage])
            char_sim = cosine_similarity(query_char, passage_char)[0, 0]
            features['char_ngram_similarity'] = char_sim
            
            # Word n-gram similarity
            query_word = self.word_ngram.transform([query])
            passage_word = self.word_ngram.transform([passage])
            word_sim = cosine_similarity(query_word, passage_word)[0, 0]
            features['word_ngram_similarity'] = word_sim
            
            # Combined similarity
            features['combined_similarity'] = (tfidf_sim + char_sim + word_sim) / 3
            
        except Exception as e:
            logger.warning(f"Error extracting semantic features: {e}")
            features.update({
                'tfidf_similarity': 0.0,
                'char_ngram_similarity': 0.0,
                'word_ngram_similarity': 0.0,
                'combined_similarity': 0.0
            })
        
        return features
    
    def extract_positional_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract advanced positional features."""
        features = {}
        
        query_lower = query.lower()
        passage_lower = passage.lower()
        
        # Exact substring match
        if query_lower in passage_lower:
            position = passage_lower.find(query_lower)
            features['exact_match'] = 1.0
            features['match_position'] = position / max(len(passage), 1)
            features['match_position_normalized'] = position / max(len(passage) - len(query), 1)
        else:
            features['exact_match'] = 0.0
            features['match_position'] = 0.0
            features['match_position_normalized'] = 0.0
        
        # Subsequence analysis
        query_words = query_lower.split()
        passage_words = passage_lower.split()
        
        if query_words and passage_words:
            # Longest common subsequence
            lcs_length = self._longest_common_subsequence(query_words, passage_words)
            features['word_lcs_ratio'] = lcs_length / len(query_words)
            
            # Consecutive word matches
            max_consecutive = self._max_consecutive_matches(query_words, passage_words)
            features['max_consecutive_matches'] = max_consecutive / len(query_words)
            
            # First/last word matches
            features['first_word_match'] = float(query_words[0] == passage_words[0])
            features['last_word_match'] = float(query_words[-1] == passage_words[-1])
            
            # Order preservation
            features['order_preservation'] = self._calculate_order_preservation(query_words, passage_words)
        else:
            features.update({
                'word_lcs_ratio': 0.0,
                'max_consecutive_matches': 0.0,
                'first_word_match': 0.0,
                'last_word_match': 0.0,
                'order_preservation': 0.0
            })
        
        return features
    
    def extract_sanskrit_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract Sanskrit-specific features."""
        features = {}
        
        # Pattern matching
        for pattern_name, pattern in self.sanskrit_patterns.items():
            query_matches = len(re.findall(pattern, query.lower()))
            passage_matches = len(re.findall(pattern, passage.lower()))
            
            features[f'query_{pattern_name}_count'] = query_matches
            features[f'passage_{pattern_name}_count'] = passage_matches
            features[f'{pattern_name}_overlap'] = min(query_matches, passage_matches)
        
        # Diacritic analysis
        diacritics = 'āīūṛṝḷḹēōṃḥṅñṭḍṇśṣ'
        query_diacritics = sum(1 for c in query if c in diacritics)
        passage_diacritics = sum(1 for c in passage if c in diacritics)
        
        features['query_diacritic_density'] = query_diacritics / max(len(query), 1)
        features['passage_diacritic_density'] = passage_diacritics / max(len(passage), 1)
        features['diacritic_density_diff'] = abs(features['query_diacritic_density'] - features['passage_diacritic_density'])
        
        # Sanskrit word endings (common suffixes)
        sanskrit_endings = ['ati', 'anti', 'aḥ', 'āḥ', 'am', 'ān', 'ena', 'aiḥ']
        query_endings = sum(1 for ending in sanskrit_endings if query.lower().endswith(ending))
        passage_endings = sum(1 for ending in sanskrit_endings if passage.lower().endswith(ending))
        
        features['query_sanskrit_endings'] = query_endings
        features['passage_sanskrit_endings'] = passage_endings
        
        return features
    
    def _longest_common_subsequence(self, seq1: List[str], seq2: List[str]) -> int:
        """Calculate longest common subsequence length."""
        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def _max_consecutive_matches(self, seq1: List[str], seq2: List[str]) -> int:
        """Find maximum consecutive word matches."""
        max_consecutive = 0
        
        for i in range(len(seq1)):
            for j in range(len(seq2)):
                consecutive = 0
                k = 0
                while (i + k < len(seq1) and j + k < len(seq2) and 
                       seq1[i + k] == seq2[j + k]):
                    consecutive += 1
                    k += 1
                max_consecutive = max(max_consecutive, consecutive)
        
        return max_consecutive
    
    def _calculate_order_preservation(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate how well word order is preserved."""
        if not seq1 or not seq2:
            return 0.0
        
        # Find positions of seq1 words in seq2
        positions = []
        for word in seq1:
            try:
                pos = seq2.index(word)
                positions.append(pos)
            except ValueError:
                continue
        
        if len(positions) < 2:
            return float(len(positions) > 0)
        
        # Check if positions are in ascending order
        ascending_pairs = sum(1 for i in range(len(positions)-1) 
                            if positions[i] < positions[i+1])
        
        return ascending_pairs / (len(positions) - 1)
    
    def extract_all_features(self, query: str, passage: str) -> Dict[str, float]:
        """Extract all features for a query-passage pair."""
        features = {}
        
        features.update(self.extract_lexical_features(query, passage))
        features.update(self.extract_semantic_features(query, passage))
        features.update(self.extract_positional_features(query, passage))
        features.update(self.extract_sanskrit_features(query, passage))
        
        return features

class XGBoostCrossEncoder:
    """Advanced cross-encoder using XGBoost for ranking."""
    
    def __init__(self, model_type: str = 'auto'):
        self.model_type = model_type
        self.feature_extractor = AdvancedFeatureExtractor()
        self.model = None
        self.feature_names = None
        self.is_trained = False
        
        # Initialize model based on availability
        if model_type == 'auto':
            if USE_XGBOOST:
                self.model_type = 'xgboost'
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                )
            elif USE_LIGHTGBM:
                self.model_type = 'lightgbm'
                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=6,
                    learning_rate=0.1,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                )
            else:
                self.model_type = 'random_forest'
                from sklearn.ensemble import RandomForestClassifier
                self.model = RandomForestClassifier(
                    n_estimators=200,
                    max_depth=10,
                    random_state=42
                )
        
        logger.info(f"Initialized {self.model_type} cross-encoder")
    
    def create_training_data_from_csv(self, csv_path: str) -> Tuple[List[Dict], List[int]]:
        """Create training data from large CSV dataset."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loading training data from {csv_path}: {len(df)} pairs")
        
        # Fit feature extractors on all texts
        all_texts = list(df['query']) + list(df['passage'])
        self.feature_extractor.fit(all_texts)
        
        training_pairs = []
        labels = []
        
        for _, row in df.iterrows():
            query = str(row['query'])
            passage = str(row['passage'])
            label = int(row['label'])
            
            features = self.feature_extractor.extract_all_features(query, passage)
            training_pairs.append(features)
            labels.append(label)
        
        logger.info(f"Created {len(training_pairs)} training pairs with {len(training_pairs[0])} features each")
        return training_pairs, labels
    
    def train(self, training_pairs: List[Dict], labels: List[int], validation_split: float = 0.2):
        """Train the XGBoost model with validation."""
        logger.info(f"Training {self.model_type} with {len(training_pairs)} pairs")
        
        if not training_pairs:
            raise ValueError("No training pairs provided")
        
        # Convert features to matrix
        self.feature_names = sorted(training_pairs[0].keys())
        X = np.array([[pair.get(feature, 0.0) for feature in self.feature_names] 
                     for pair in training_pairs])
        y = np.array(labels)
        
        # Train-validation split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Train model
        if USE_XGBOOST and self.model_type == 'xgboost':
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(X_train)
        val_pred = self.model.predict(X_val)
        
        train_acc = accuracy_score(y_train, train_pred)
        val_acc = accuracy_score(y_val, val_pred)
        
        # Calculate AUC if probabilities available
        try:
            train_proba = self.model.predict_proba(X_train)[:, 1]
            val_proba = self.model.predict_proba(X_val)[:, 1]
            train_auc = roc_auc_score(y_train, train_proba)
            val_auc = roc_auc_score(y_val, val_proba)
            
            logger.info(f"Training - Accuracy: {train_acc:.3f}, AUC: {train_auc:.3f}")
            logger.info(f"Validation - Accuracy: {val_acc:.3f}, AUC: {val_auc:.3f}")
        except:
            logger.info(f"Training accuracy: {train_acc:.3f}")
            logger.info(f"Validation accuracy: {val_acc:.3f}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
            
            logger.info("Top 10 most important features:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.3f}")
        
        return val_acc
    
    def rank_passages(self, query: str, passages: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Rank passages using trained model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        scored_passages = []
        
        for passage_data in passages:
            passage_text = passage_data.get('text', '')
            
            # Extract features
            features = self.feature_extractor.extract_all_features(query, passage_text)
            
            # Convert to feature vector
            feature_vector = np.array([[features.get(feature, 0.0) for feature in self.feature_names]])
            
            # Get score
            try:
                score = self.model.predict_proba(feature_vector)[0, 1]
            except:
                score = self.model.decision_function(feature_vector)[0] if hasattr(self.model, 'decision_function') else 0.0
            
            # Add score to passage data
            scored_passage = passage_data.copy()
            scored_passage['ranking_score'] = float(score)
            scored_passages.append(scored_passage)
        
        # Sort by score descending
        scored_passages.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        if top_k:
            scored_passages = scored_passages[:top_k]
        
        # Add ranking positions
        for i, passage in enumerate(scored_passages):
            passage['final_rank'] = i + 1
        
        return scored_passages

def main():
    """Test XGBoost cross-encoder."""
    logging.basicConfig(level=logging.INFO)
    
    # Check which model we're using
    print(f"🚀 XGBoost Cross-Encoder Demo")
    print(f"Using: {('XGBoost' if USE_XGBOOST else 'LightGBM' if USE_LIGHTGBM else 'Random Forest')}")
    
    # Initialize model
    ranker = XGBoostCrossEncoder()
    
    # Test with sample data
    csv_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    
    if not Path(csv_path).exists():
        print("❌ Training data not found. Run data_generation.py first.")
        return
    
    # Create simple training data for demo
    df = pd.read_csv(csv_path)
    
    # Create positive and negative pairs
    training_data = []
    labels = []
    
    for _, row in df.head(50).iterrows():  # Small demo dataset
        text = str(row['text'])
        words = text.split()
        
        if len(words) >= 4:
            # Positive pair
            query = ' '.join(words[:len(words)//2])
            training_data.append({'query': query, 'passage': text})
            labels.append(1)
            
            # Negative pair
            neg_text = df.sample(1).iloc[0]['text']
            if str(neg_text) != text:
                training_data.append({'query': query, 'passage': str(neg_text)})
                labels.append(0)
    
    # Convert to feature format
    all_texts = [d['query'] for d in training_data] + [d['passage'] for d in training_data]
    ranker.feature_extractor.fit(all_texts)
    
    feature_pairs = []
    for data in training_data:
        features = ranker.feature_extractor.extract_all_features(data['query'], data['passage'])
        feature_pairs.append(features)
    
    # Train model
    print(f"\n📊 Training with {len(feature_pairs)} pairs...")
    accuracy = ranker.train(feature_pairs, labels)
    
    # Test ranking
    test_query = "dharma viṣṇu namaḥ"
    test_passages = [
        {'text': 'dharmaviṣṇupriya śrīkṛṣṇa namaḥ paramātma sanātana', 'id': 1},
        {'text': 'rājya dhana āryāṇām viṣṇave namaḥ', 'id': 2},  
        {'text': 'gaṅgā yamunā sarasvatī devī pārvatī', 'id': 3},
        {'text': 'dharma artha kāma mokṣa viṣṇu namaḥ', 'id': 4}
    ]
    
    ranked = ranker.rank_passages(test_query, test_passages)
    
    print(f"\n🎯 Ranking Results for: '{test_query}'")
    print("-" * 50)
    
    for passage in ranked:
        print(f"Rank {passage['final_rank']}: Score={passage['ranking_score']:.3f}")
        print(f"  Text: {passage['text']}")
        print(f"  ID: {passage['id']}")

if __name__ == "__main__":
    main()