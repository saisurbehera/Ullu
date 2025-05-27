import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
import json
import random
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, ndcg_score
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)

class CrossEncoderTestDataGenerator:
    """Generate high-quality test datasets for cross-encoder evaluation."""
    
    def __init__(self, corpus_path: str):
        self.corpus_df = pd.read_csv(corpus_path)
        self.test_datasets = {}
    
    def create_exact_match_test(self, num_queries: int = 200) -> pd.DataFrame:
        """Test dataset with exact substring matches (should score highest)."""
        test_pairs = []
        
        sampled_passages = self.corpus_df.sample(n=num_queries, random_state=42)
        
        for idx, row in sampled_passages.iterrows():
            text = str(row['text'])
            words = text.split()
            
            if len(words) >= 6:
                # Create exact substring queries of different lengths
                for ratio in [0.3, 0.5, 0.7]:
                    query_len = max(2, int(len(words) * ratio))
                    start_pos = random.randint(0, len(words) - query_len)
                    query = ' '.join(words[start_pos:start_pos + query_len])
                    
                    test_pairs.append({
                        'query': query,
                        'passage': text,
                        'label': 1,
                        'test_type': 'exact_match',
                        'expected_score': 'high',
                        'difficulty': 'easy',
                        'work': row.get('work', 'unknown'),
                        'category': row.get('category', 'unknown'),
                        'query_coverage': ratio
                    })
        
        return pd.DataFrame(test_pairs)
    
    def create_partial_match_test(self, num_queries: int = 200) -> pd.DataFrame:
        """Test dataset with partial word overlaps (medium scores)."""
        test_pairs = []
        
        sampled_passages = self.corpus_df.sample(n=num_queries, random_state=43)
        
        for idx, row in sampled_passages.iterrows():
            text = str(row['text'])
            words = text.split()
            
            if len(words) >= 8:
                # Create queries with partial overlap
                for overlap_ratio in [0.3, 0.5, 0.7]:
                    num_overlapping = max(2, int(len(words) * overlap_ratio))
                    
                    # Take some words from the passage
                    overlapping_words = random.sample(words, min(num_overlapping, len(words)))
                    
                    # Add some random words from other passages
                    other_words = []
                    for _ in range(random.randint(1, 3)):
                        other_passage = self.corpus_df.sample(1).iloc[0]['text']
                        other_word = random.choice(str(other_passage).split())
                        if other_word not in overlapping_words:
                            other_words.append(other_word)
                    
                    # Combine words
                    query_words = overlapping_words + other_words
                    random.shuffle(query_words)
                    query = ' '.join(query_words[:6])  # Limit query length
                    
                    test_pairs.append({
                        'query': query,
                        'passage': text,
                        'label': 1,
                        'test_type': 'partial_match',
                        'expected_score': 'medium',
                        'difficulty': 'medium',
                        'work': row.get('work', 'unknown'),
                        'category': row.get('category', 'unknown'),
                        'overlap_ratio': overlap_ratio
                    })
        
        return pd.DataFrame(test_pairs)
    
    def create_semantic_match_test(self, num_queries: int = 150) -> pd.DataFrame:
        """Test dataset with semantic similarity (Sanskrit synonyms/epithets)."""
        test_pairs = []
        
        # Sanskrit synonym pairs
        synonyms = {
            'viṣṇu': ['hari', 'nārāyaṇa', 'keśava', 'madhusūdana', 'govinda', 'kṛṣṇa'],
            'śiva': ['maheśvara', 'śaṅkara', 'rudra', 'īśa', 'hara', 'śambhu'],
            'brahmā': ['pitāmaha', 'svayambhū', 'prajāpati', 'virañci', 'caturmukha'],
            'devī': ['ambā', 'umā', 'pārvatī', 'śakti', 'bhagavatī', 'mātā'],
            'dharma': ['nyāya', 'satya', 'rita', 'yuga'],
            'mokṣa': ['mukti', 'nirvāṇa', 'kaivalya', 'apavarga'],
            'guru': ['ācārya', 'śikṣaka', 'upādhyāya'],
            'rāja': ['nṛpa', 'bhūpati', 'bhūpāla', 'mahīpati']
        }
        
        for original, synonym_list in synonyms.items():
            # Find passages containing the original term
            matching_passages = self.corpus_df[
                self.corpus_df['text'].str.contains(original, case=False, na=False)
            ]
            
            if len(matching_passages) > 0:
                for _, row in matching_passages.sample(n=min(len(matching_passages), 20)).iterrows():
                    text = str(row['text'])
                    
                    # Create query using synonyms
                    for synonym in synonym_list[:3]:  # Use first 3 synonyms
                        # Replace original with synonym in a portion of the text
                        words = text.lower().split()
                        query_words = []
                        
                        for word in words[:6]:  # Take first 6 words
                            if original in word:
                                query_words.append(word.replace(original, synonym))
                            else:
                                query_words.append(word)
                        
                        if len(query_words) >= 3:
                            query = ' '.join(query_words)
                            
                            test_pairs.append({
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
        
        return pd.DataFrame(test_pairs)
    
    def create_negative_test(self, positive_df: pd.DataFrame, negatives_per_positive: int = 2) -> pd.DataFrame:
        """Create negative test pairs with different difficulty levels."""
        negative_pairs = []
        
        for _, pos_row in positive_df.iterrows():
            query = pos_row['query']
            pos_work = pos_row['work']
            pos_category = pos_row['category']
            
            # Create different types of negatives
            for neg_type in ['random', 'same_category', 'same_work']:
                try:
                    if neg_type == 'random':
                        # Completely random passage
                        neg_passage = self.corpus_df.sample(1).iloc[0]
                        expected_score = 'very_low'
                        difficulty = 'easy'
                        
                    elif neg_type == 'same_category':
                        # Same category, different work
                        candidates = self.corpus_df[
                            (self.corpus_df['category'] == pos_category) & 
                            (self.corpus_df['work'] != pos_work)
                        ]
                        if len(candidates) > 0:
                            neg_passage = candidates.sample(1).iloc[0]
                            expected_score = 'low'
                            difficulty = 'medium'
                        else:
                            continue
                            
                    elif neg_type == 'same_work':
                        # Same work, different passage
                        candidates = self.corpus_df[
                            (self.corpus_df['work'] == pos_work) & 
                            (self.corpus_df['text'] != pos_row['passage'])
                        ]
                        if len(candidates) > 0:
                            neg_passage = candidates.sample(1).iloc[0]
                            expected_score = 'low-medium'
                            difficulty = 'hard'
                        else:
                            continue
                    
                    negative_pairs.append({
                        'query': query,
                        'passage': str(neg_passage['text']),
                        'label': 0,
                        'test_type': f'negative_{neg_type}',
                        'expected_score': expected_score,
                        'difficulty': difficulty,
                        'work': neg_passage.get('work', 'unknown'),
                        'category': neg_passage.get('category', 'unknown'),
                        'negative_type': neg_type
                    })
                    
                except Exception as e:
                    logger.warning(f"Error creating negative pair: {e}")
                    continue
        
        return pd.DataFrame(negative_pairs)
    
    def create_comprehensive_test_dataset(self, save_path: Optional[str] = None) -> pd.DataFrame:
        """Create comprehensive test dataset with all test types."""
        logger.info("Creating comprehensive cross-encoder test dataset...")
        
        # Create different test types
        exact_test = self.create_exact_match_test(200)
        logger.info(f"Created {len(exact_test)} exact match tests")
        
        partial_test = self.create_partial_match_test(200)
        logger.info(f"Created {len(partial_test)} partial match tests")
        
        semantic_test = self.create_semantic_match_test(150)
        logger.info(f"Created {len(semantic_test)} semantic match tests")
        
        # Combine positive tests
        positive_tests = pd.concat([exact_test, partial_test, semantic_test], ignore_index=True)
        
        # Create negative tests
        negative_test = self.create_negative_test(positive_tests, negatives_per_positive=1)
        logger.info(f"Created {len(negative_test)} negative tests")
        
        # Combine all tests
        complete_test = pd.concat([positive_tests, negative_test], ignore_index=True)
        complete_test = complete_test.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle
        
        # Add test metadata
        complete_test['test_id'] = range(len(complete_test))
        
        logger.info(f"Complete test dataset: {len(complete_test)} pairs")
        logger.info(f"Positive: {len(complete_test[complete_test['label'] == 1])}")
        logger.info(f"Negative: {len(complete_test[complete_test['label'] == 0])}")
        
        # Show distribution
        logger.info("Test type distribution:")
        for test_type, count in complete_test['test_type'].value_counts().items():
            logger.info(f"  {test_type}: {count}")
        
        if save_path:
            complete_test.to_csv(save_path, index=False)
            logger.info(f"Test dataset saved to {save_path}")
        
        return complete_test

class CrossEncoderEvaluator:
    """Comprehensive evaluation framework for cross-encoder models."""
    
    def __init__(self):
        self.results = {}
    
    def evaluate_model(self, model, test_df: pd.DataFrame, model_name: str = "model") -> Dict:
        """Evaluate a cross-encoder model on test dataset."""
        logger.info(f"Evaluating {model_name} on {len(test_df)} test pairs...")
        
        if not hasattr(model, 'feature_extractor') or not model.feature_extractor.is_fitted:
            # Fit feature extractor if needed
            all_texts = list(test_df['query']) + list(test_df['passage'])
            model.feature_extractor.fit(all_texts)
        
        predictions = []
        probabilities = []
        
        for _, row in test_df.iterrows():
            query = str(row['query'])
            passage = str(row['passage'])
            
            # Extract features
            features = model.feature_extractor.extract_all_features(query, passage)
            
            # Convert to feature vector
            if not hasattr(model, 'feature_names') or model.feature_names is None:
                model.feature_names = sorted(features.keys())
            
            feature_vector = np.array([[features.get(feat, 0.0) for feat in model.feature_names]])
            
            # Get prediction and probability
            pred = model.model.predict(feature_vector)[0]
            
            try:
                if hasattr(model.model, 'predict_proba'):
                    prob = model.model.predict_proba(feature_vector)[0, 1]
                elif hasattr(model.model, 'predict'):
                    # For LTR models, use raw score
                    prob = model.model.predict(feature_vector)[0] if hasattr(model, 'model') else 0.5
                else:
                    prob = 0.5
            except:
                prob = pred  # Fallback to binary prediction
            
            predictions.append(pred)
            probabilities.append(prob)
        
        # Calculate metrics
        labels = test_df['label'].values
        
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='binary')
        
        try:
            auc = roc_auc_score(labels, probabilities)
        except:
            auc = 0.5
        
        # Calculate metrics by test type
        type_metrics = {}
        for test_type in test_df['test_type'].unique():
            type_mask = test_df['test_type'] == test_type
            type_labels = labels[type_mask]
            type_preds = np.array(predictions)[type_mask]
            
            if len(type_labels) > 0:
                type_acc = accuracy_score(type_labels, type_preds)
                type_metrics[test_type] = {
                    'accuracy': type_acc,
                    'count': len(type_labels),
                    'positive_ratio': np.mean(type_labels)
                }
        
        # Calculate metrics by difficulty
        difficulty_metrics = {}
        for difficulty in test_df['difficulty'].unique():
            diff_mask = test_df['difficulty'] == difficulty
            diff_labels = labels[diff_mask]
            diff_preds = np.array(predictions)[diff_mask]
            
            if len(diff_labels) > 0:
                diff_acc = accuracy_score(diff_labels, diff_preds)
                difficulty_metrics[difficulty] = {
                    'accuracy': diff_acc,
                    'count': len(diff_labels)
                }
        
        # Ranking evaluation (for positive pairs only)
        ranking_metrics = self._evaluate_ranking_quality(test_df, probabilities)
        
        results = {
            'model_name': model_name,
            'overall_metrics': {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'auc': auc,
                'total_pairs': len(test_df)
            },
            'type_metrics': type_metrics,
            'difficulty_metrics': difficulty_metrics,
            'ranking_metrics': ranking_metrics
        }
        
        self.results[model_name] = results
        
        logger.info(f"{model_name} Results:")
        logger.info(f"  Overall Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1: {f1:.3f}")
        logger.info(f"  AUC: {auc:.3f}")
        
        return results
    
    def _evaluate_ranking_quality(self, test_df: pd.DataFrame, probabilities: List[float]) -> Dict:
        """Evaluate ranking quality for queries with multiple candidates."""
        ranking_metrics = {}
        
        # Group by query
        test_df_with_probs = test_df.copy()
        test_df_with_probs['probability'] = probabilities
        
        grouped = test_df_with_probs.groupby('query')
        
        ndcg_scores = []
        mrr_scores = []
        precision_at_1_scores = []
        
        for query, group in grouped:
            if len(group) < 2:  # Need at least 2 candidates for ranking
                continue
            
            # Sort by probability (descending)
            sorted_group = group.sort_values('probability', ascending=False)
            
            labels = sorted_group['label'].values
            
            # NDCG@5
            if len(labels) > 1:
                ndcg = ndcg_score([labels], [sorted_group['probability'].values], k=min(5, len(labels)))
                ndcg_scores.append(ndcg)
            
            # MRR (Mean Reciprocal Rank)
            for i, label in enumerate(labels):
                if label == 1:  # First relevant result
                    mrr_scores.append(1.0 / (i + 1))
                    break
            else:
                mrr_scores.append(0.0)
            
            # Precision@1
            precision_at_1_scores.append(float(labels[0] == 1))
        
        ranking_metrics = {
            'ndcg_at_5': np.mean(ndcg_scores) if ndcg_scores else 0.0,
            'mrr': np.mean(mrr_scores) if mrr_scores else 0.0,
            'precision_at_1': np.mean(precision_at_1_scores) if precision_at_1_scores else 0.0,
            'ranking_queries': len(ndcg_scores)
        }
        
        return ranking_metrics
    
    def compare_models(self, results_dict: Dict) -> pd.DataFrame:
        """Compare multiple model results."""
        comparison_data = []
        
        for model_name, results in results_dict.items():
            metrics = results['overall_metrics']
            ranking = results['ranking_metrics']
            
            comparison_data.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Precision': metrics['precision'],
                'Recall': metrics['recall'],
                'F1': metrics['f1'],
                'AUC': metrics['auc'],
                'NDCG@5': ranking['ndcg_at_5'],
                'MRR': ranking['mrr'],
                'P@1': ranking['precision_at_1']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        return comparison_df.round(4)
    
    def plot_evaluation_results(self, save_path: Optional[str] = None):
        """Create visualization of evaluation results."""
        if not self.results:
            logger.warning("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Overall metrics comparison
        comparison_df = self.compare_models(self.results)
        
        # Plot 1: Overall accuracy by model
        axes[0, 0].bar(comparison_df['Model'], comparison_df['Accuracy'])
        axes[0, 0].set_title('Overall Accuracy by Model')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Plot 2: Precision-Recall
        axes[0, 1].scatter(comparison_df['Recall'], comparison_df['Precision'], s=100)
        for i, model in enumerate(comparison_df['Model']):
            axes[0, 1].annotate(model, (comparison_df['Recall'].iloc[i], comparison_df['Precision'].iloc[i]))
        axes[0, 1].set_xlabel('Recall')
        axes[0, 1].set_ylabel('Precision')
        axes[0, 1].set_title('Precision vs Recall')
        
        # Plot 3: Ranking metrics
        ranking_metrics = ['NDCG@5', 'MRR', 'P@1']
        x = np.arange(len(comparison_df))
        width = 0.25
        
        for i, metric in enumerate(ranking_metrics):
            axes[1, 0].bar(x + i*width, comparison_df[metric], width, label=metric)
        
        axes[1, 0].set_xlabel('Models')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_title('Ranking Metrics Comparison')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(comparison_df['Model'], rotation=45)
        axes[1, 0].legend()
        
        # Plot 4: Performance by difficulty
        if len(self.results) > 0:
            first_result = list(self.results.values())[0]
            difficulty_data = first_result['difficulty_metrics']
            
            difficulties = list(difficulty_data.keys())
            accuracies = [difficulty_data[d]['accuracy'] for d in difficulties]
            
            axes[1, 1].bar(difficulties, accuracies)
            axes[1, 1].set_title('Accuracy by Difficulty Level')
            axes[1, 1].set_ylabel('Accuracy')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        plt.show()

def main():
    """Run comprehensive cross-encoder evaluation."""
    logging.basicConfig(level=logging.INFO)
    
    corpus_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    test_dataset_path = "/home/sai/Desktop/ullu/data/cross_encoder_test_dataset.csv"
    
    if not Path(corpus_path).exists():
        print("❌ Corpus not found. Run data processing first.")
        return
    
    print("🧪 Cross-Encoder Evaluation Framework")
    print("=" * 50)
    
    # Generate test dataset
    generator = CrossEncoderTestDataGenerator(corpus_path)
    test_df = generator.create_comprehensive_test_dataset(test_dataset_path)
    
    print(f"\n📊 Test Dataset Created:")
    print(f"Total pairs: {len(test_df)}")
    print(f"Positive pairs: {len(test_df[test_df['label'] == 1])}")
    print(f"Negative pairs: {len(test_df[test_df['label'] == 0])}")
    
    print(f"\nTest type distribution:")
    for test_type, count in test_df['test_type'].value_counts().items():
        print(f"  {test_type}: {count}")
    
    print(f"\nDifficulty distribution:")
    for difficulty, count in test_df['difficulty'].value_counts().items():
        print(f"  {difficulty}: {count}")
    
    # Show sample test cases
    print(f"\n📝 Sample Test Cases:")
    for test_type in ['exact_match', 'partial_match', 'semantic_match', 'negative_random']:
        sample = test_df[test_df['test_type'] == test_type].iloc[0]
        print(f"\n{test_type.upper()}:")
        print(f"  Query: {sample['query']}")
        print(f"  Passage: {sample['passage'][:80]}...")
        print(f"  Label: {sample['label']} (Expected: {sample['expected_score']})")

if __name__ == "__main__":
    main()