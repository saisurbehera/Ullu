import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import logging
import pickle
from pathlib import Path
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ndcg_score

# Import the advanced feature extractor
from ranking.xgboost_cross_encoder import AdvancedFeatureExtractor

try:
    import xgboost as xgb
    USE_XGBOOST = True
except ImportError:
    USE_XGBOOST = False
    print("❌ XGBoost not available. Install with: pip install xgboost")

logger = logging.getLogger(__name__)

class XGBoostLTRRanker:
    """
    XGBoost Learning-to-Rank (LTR) implementation for Sanskrit quote ranking.
    Uses ranking objectives like rank:pairwise, rank:ndcg, or rank:map.
    """
    
    def __init__(self, 
                 objective: str = 'rank:pairwise',
                 eval_metric: str = 'ndcg',
                 n_estimators: int = 300,
                 max_depth: int = 6,
                 learning_rate: float = 0.1):
        
        if not USE_XGBOOST:
            raise ImportError("XGBoost is required for LTR ranking")
        
        self.objective = objective
        self.eval_metric = eval_metric
        self.feature_extractor = AdvancedFeatureExtractor()
        self.feature_names = None
        self.is_trained = False
        
        # XGBoost LTR parameters
        self.params = {
            'objective': objective,
            'eval_metric': eval_metric,
            'eta': learning_rate,  # learning rate
            'max_depth': max_depth,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'silent': 1,
            'seed': 42
        }
        
        self.num_boost_round = n_estimators
        self.model = None
        
        logger.info(f"Initialized XGBoost LTR with objective: {objective}")
    
    def prepare_ltr_data(self, queries: List[str], passages_per_query: List[List[Dict]], 
                        relevance_scores: List[List[int]]) -> Tuple[xgb.DMatrix, List[int]]:
        """
        Prepare data in LTR format for XGBoost.
        
        Args:
            queries: List of query strings
            passages_per_query: List of passage lists for each query
            relevance_scores: List of relevance score lists (0=irrelevant, 1=relevant, 2=highly_relevant)
            
        Returns:
            XGBoost DMatrix and group sizes
        """
        all_features = []
        all_labels = []
        group_sizes = []
        
        # Fit feature extractor on all texts
        all_texts = queries.copy()
        for passages in passages_per_query:
            all_texts.extend([p.get('text', '') for p in passages])
        
        if not self.feature_extractor.is_fitted:
            self.feature_extractor.fit(all_texts)
        
        for i, (query, passages, scores) in enumerate(zip(queries, passages_per_query, relevance_scores)):
            query_features = []
            query_labels = []
            
            for passage, score in zip(passages, scores):
                passage_text = passage.get('text', '')
                
                # Extract features
                features = self.feature_extractor.extract_all_features(query, passage_text)
                
                # Convert to feature vector
                if self.feature_names is None:
                    self.feature_names = sorted(features.keys())
                
                feature_vector = [features.get(feat, 0.0) for feat in self.feature_names]
                
                query_features.append(feature_vector)
                query_labels.append(score)
            
            all_features.extend(query_features)
            all_labels.extend(query_labels)
            group_sizes.append(len(query_features))
        
        # Create DMatrix with group information
        dtrain = xgb.DMatrix(np.array(all_features), label=np.array(all_labels))
        dtrain.set_group(group_sizes)
        
        return dtrain, group_sizes
    
    def create_ltr_training_data(self, df: pd.DataFrame, queries_per_group: int = 20) -> Tuple[List[str], List[List[Dict]], List[List[int]]]:
        """
        Create LTR training data from query-passage pairs DataFrame.
        Groups passages by query and assigns relevance scores.
        """
        # Group by query
        grouped = df.groupby('query')
        
        queries = []
        passages_per_query = []
        relevance_scores = []
        
        for query, group in grouped:
            if len(group) < 2:  # Need at least 2 passages per query
                continue
            
            query_passages = []
            query_scores = []
            
            for _, row in group.iterrows():
                passage_data = {
                    'text': row['passage'],
                    'work': row.get('work', 'unknown'),
                    'category': row.get('category', 'unknown')
                }
                
                # Relevance score based on label and difficulty
                if row['label'] == 1:  # Positive
                    if row.get('difficulty', 'medium') == 'easy':
                        score = 2  # Highly relevant
                    else:
                        score = 1  # Relevant
                else:  # Negative
                    score = 0  # Irrelevant
                
                query_passages.append(passage_data)
                query_scores.append(score)
            
            queries.append(query)
            passages_per_query.append(query_passages)
            relevance_scores.append(query_scores)
        
        logger.info(f"Created LTR data: {len(queries)} queries with avg {np.mean([len(p) for p in passages_per_query]):.1f} passages each")
        
        return queries, passages_per_query, relevance_scores
    
    def train_from_csv(self, csv_path: str, validation_split: float = 0.2):
        """Train LTR model from CSV dataset."""
        df = pd.read_csv(csv_path)
        logger.info(f"Loading LTR training data from {csv_path}: {len(df)} pairs")
        
        # Create LTR format data
        queries, passages_per_query, relevance_scores = self.create_ltr_training_data(df)
        
        # Split into train/validation
        split_idx = int(len(queries) * (1 - validation_split))
        
        train_queries = queries[:split_idx]
        train_passages = passages_per_query[:split_idx]
        train_scores = relevance_scores[:split_idx]
        
        val_queries = queries[split_idx:]
        val_passages = passages_per_query[split_idx:]
        val_scores = relevance_scores[split_idx:]
        
        # Prepare XGBoost data
        dtrain, train_groups = self.prepare_ltr_data(train_queries, train_passages, train_scores)
        dval, val_groups = self.prepare_ltr_data(val_queries, val_passages, val_scores)
        
        # Train model
        evallist = [(dtrain, 'train'), (dval, 'eval')]
        
        logger.info(f"Training XGBoost LTR model...")
        logger.info(f"Training queries: {len(train_queries)}, Validation queries: {len(val_queries)}")
        
        self.model = xgb.train(
            self.params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evallist,
            early_stopping_rounds=50,
            verbose_eval=False
        )
        
        self.is_trained = True
        
        # Evaluate
        train_pred = self.model.predict(dtrain)
        val_pred = self.model.predict(dval)
        
        # Calculate NDCG for train and validation
        train_ndcg = self._calculate_group_ndcg(train_pred, dtrain.get_label(), train_groups)
        val_ndcg = self._calculate_group_ndcg(val_pred, dval.get_label(), val_groups)
        
        logger.info(f"Training NDCG@5: {train_ndcg:.4f}")
        logger.info(f"Validation NDCG@5: {val_ndcg:.4f}")
        
        # Feature importance
        importance = self.model.get_score(importance_type='weight')
        if importance:
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            logger.info("Top 10 most important features:")
            for feat, score in sorted_importance[:10]:
                feat_name = self.feature_names[int(feat[1:])] if feat.startswith('f') else feat
                logger.info(f"  {feat_name}: {score}")
        
        return val_ndcg
    
    def _calculate_group_ndcg(self, predictions: np.ndarray, labels: np.ndarray, 
                             group_sizes: List[int], k: int = 5) -> float:
        """Calculate NDCG@k for grouped predictions."""
        ndcg_scores = []
        start_idx = 0
        
        for group_size in group_sizes:
            end_idx = start_idx + group_size
            
            group_pred = predictions[start_idx:end_idx]
            group_labels = labels[start_idx:end_idx]
            
            if len(group_pred) > 1:
                # Reshape for ndcg_score function
                ndcg = ndcg_score([group_labels], [group_pred], k=min(k, len(group_pred)))
                ndcg_scores.append(ndcg)
            
            start_idx = end_idx
        
        return np.mean(ndcg_scores) if ndcg_scores else 0.0
    
    def rank_passages(self, query: str, passages: List[Dict], top_k: Optional[int] = None) -> List[Dict]:
        """Rank passages using trained LTR model."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train_from_csv() first.")
        
        if not passages:
            return []
        
        # Extract features for all passages
        features_list = []
        for passage in passages:
            passage_text = passage.get('text', '')
            features = self.feature_extractor.extract_all_features(query, passage_text)
            feature_vector = [features.get(feat, 0.0) for feat in self.feature_names]
            features_list.append(feature_vector)
        
        # Create DMatrix
        dtest = xgb.DMatrix(np.array(features_list))
        
        # Get predictions
        scores = self.model.predict(dtest)
        
        # Add scores to passages and sort
        scored_passages = []
        for i, passage in enumerate(passages):
            scored_passage = passage.copy()
            scored_passage['ltr_score'] = float(scores[i])
            scored_passages.append(scored_passage)
        
        # Sort by LTR score descending
        scored_passages.sort(key=lambda x: x['ltr_score'], reverse=True)
        
        if top_k:
            scored_passages = scored_passages[:top_k]
        
        # Add ranking positions
        for i, passage in enumerate(scored_passages):
            passage['final_rank'] = i + 1
        
        return scored_passages
    
    def save(self, path: str):
        """Save the trained LTR model."""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'feature_extractor': self.feature_extractor,
            'feature_names': self.feature_names,
            'params': self.params,
            'objective': self.objective,
            'eval_metric': self.eval_metric
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"XGBoost LTR model saved to {path}")
    
    def load(self, path: str):
        """Load a trained LTR model."""
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.feature_extractor = model_data['feature_extractor']
        self.feature_names = model_data['feature_names']
        self.params = model_data['params']
        self.objective = model_data['objective']
        self.eval_metric = model_data['eval_metric']
        self.is_trained = True
        
        logger.info(f"XGBoost LTR model loaded from {path}")

def compare_ltr_objectives():
    """Compare different LTR objectives."""
    if not USE_XGBOOST:
        print("❌ XGBoost not available")
        return
    
    objectives = [
        'rank:pairwise',   # Pairwise ranking
        'rank:ndcg',       # NDCG optimization  
        'rank:map'         # Mean Average Precision
    ]
    
    print("🎯 Comparing XGBoost LTR Objectives")
    print("=" * 50)
    
    # Create sample data
    sample_df = pd.DataFrame([
        {'query': 'dharma viṣṇu', 'passage': 'dharmaviṣṇu paramātma', 'label': 1, 'difficulty': 'easy'},
        {'query': 'dharma viṣṇu', 'passage': 'kṛṣṇa arjuna battlefield', 'label': 0, 'difficulty': 'hard'},
        {'query': 'namaḥ śivāya', 'passage': 'namaḥ śivāya oṃ', 'label': 1, 'difficulty': 'easy'},
        {'query': 'namaḥ śivāya', 'passage': 'gaṅgā yamunā sarasvatī', 'label': 0, 'difficulty': 'medium'},
        {'query': 'rāma sītā', 'passage': 'rāma sītā vana', 'label': 1, 'difficulty': 'medium'},
        {'query': 'rāma sītā', 'passage': 'viṣṇu lakṣmī vaikuṇṭha', 'label': 0, 'difficulty': 'hard'},
    ])
    
    results = {}
    
    for objective in objectives:
        print(f"\n📊 Testing {objective}...")
        
        try:
            ranker = XGBoostLTRRanker(
                objective=objective,
                n_estimators=50,  # Small for demo
                learning_rate=0.3
            )
            
            # Create temporary CSV
            temp_path = "/tmp/ltr_test.csv"
            sample_df.to_csv(temp_path, index=False)
            
            # Train
            ndcg = ranker.train_from_csv(temp_path, validation_split=0.3)
            results[objective] = ndcg
            
            print(f"   ✅ NDCG@5: {ndcg:.4f}")
            
        except Exception as e:
            print(f"   ❌ Error: {e}")
            results[objective] = 0.0
    
    print(f"\n🏆 Results Summary:")
    for obj, score in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"   {obj}: {score:.4f}")

def main():
    """Demo XGBoost LTR ranking."""
    logging.basicConfig(level=logging.INFO)
    
    if not USE_XGBOOST:
        print("❌ XGBoost not available. Install with: pip install xgboost")
        return
    
    print("🚀 XGBoost Learning-to-Rank Demo")
    print("=" * 40)
    
    # Check for training data
    large_dataset_path = "/home/sai/Desktop/ullu/data/large_training_dataset.csv"
    quotes_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    
    if Path(large_dataset_path).exists():
        training_path = large_dataset_path
        print(f"📚 Using large dataset: {training_path}")
    elif Path(quotes_path).exists():
        training_path = quotes_path
        print(f"📚 Using quotes dataset: {training_path}")
        
        # Create simple training data
        df = pd.read_csv(training_path)
        training_data = []
        
        for _, row in df.head(100).iterrows():
            text = str(row['text'])
            words = text.split()
            
            if len(words) >= 4:
                query = ' '.join(words[:len(words)//2])
                
                # Positive pair
                training_data.append({
                    'query': query,
                    'passage': text,
                    'label': 1,
                    'difficulty': 'easy',
                    'work': row.get('work', 'unknown'),
                    'category': row.get('category', 'unknown')
                })
                
                # Negative pairs
                for _ in range(2):
                    neg_row = df.sample(1).iloc[0]
                    if str(neg_row['text']) != text:
                        training_data.append({
                            'query': query,
                            'passage': str(neg_row['text']),
                            'label': 0,
                            'difficulty': 'medium',
                            'work': neg_row.get('work', 'unknown'),
                            'category': neg_row.get('category', 'unknown')
                        })
        
        # Save temporary training data
        temp_df = pd.DataFrame(training_data)
        training_path = "/tmp/ltr_training.csv"
        temp_df.to_csv(training_path, index=False)
        print(f"📝 Created training data: {len(temp_df)} pairs")
        
    else:
        print("❌ No training data found. Run data_generation.py first.")
        return
    
    # Test different LTR objectives
    objectives = ['rank:pairwise', 'rank:ndcg']
    
    for objective in objectives:
        print(f"\n🎯 Testing {objective}")
        print("-" * 30)
        
        ranker = XGBoostLTRRanker(
            objective=objective,
            eval_metric='ndcg',
            n_estimators=100,
            learning_rate=0.1
        )
        
        # Train
        ndcg = ranker.train_from_csv(training_path)
        
        # Test ranking
        test_query = "dharma viṣṇu namaḥ"
        test_passages = [
            {'text': 'dharmaviṣṇupriya śrīkṛṣṇa namaḥ paramātma', 'id': 1},
            {'text': 'rājya dhana āryāṇām viṣṇave namaḥ', 'id': 2},
            {'text': 'gaṅgā yamunā sarasvatī devī', 'id': 3},
            {'text': 'dharma artha kāma mokṣa viṣṇu namaḥ', 'id': 4}
        ]
        
        ranked = ranker.rank_passages(test_query, test_passages)
        
        print(f"\n📊 Rankings for: '{test_query}'")
        for passage in ranked:
            print(f"  Rank {passage['final_rank']}: Score={passage['ltr_score']:.3f}")
            print(f"    Text: {passage['text'][:50]}...")

if __name__ == "__main__":
    main()