import sys
sys.path.append('/home/sai/Desktop/ullu/src')

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import time

# Import all cross-encoder models
from ranking.cross_encoder import CrossEncoderRanker
from ranking.xgboost_cross_encoder import XGBoostCrossEncoder
from ranking.xgboost_ltr import XGBoostLTRRanker
from evaluation.cross_encoder_evaluation import CrossEncoderEvaluator

logger = logging.getLogger(__name__)

def prepare_training_data(corpus_path: str, size: int = 1000):
    """Prepare training data for models that need it."""
    df = pd.read_csv(corpus_path)
    
    training_data = []
    labels = []
    
    sampled = df.sample(n=min(size, len(df)), random_state=42)
    
    for _, row in sampled.iterrows():
        text = str(row['text'])
        words = text.split()
        
        if len(words) >= 4:
            # Positive pair
            query_len = max(2, len(words) // 2)
            query = ' '.join(words[:query_len])
            
            training_data.append({
                'query': query,
                'passage': text,
                'work': row.get('work', 'unknown'),
                'category': row.get('category', 'unknown')
            })
            labels.append(1)
            
            # Negative pairs
            for _ in range(2):
                neg_row = df.sample(1).iloc[0]
                if str(neg_row['text']) != text:
                    training_data.append({
                        'query': query,
                        'passage': str(neg_row['text']),
                        'work': neg_row.get('work', 'unknown'),
                        'category': neg_row.get('category', 'unknown')
                    })
                    labels.append(0)
    
    return training_data, labels

def train_and_evaluate_models(test_dataset_path: str, corpus_path: str):
    """Train and evaluate all cross-encoder models."""
    
    # Load test dataset
    test_df = pd.read_csv(test_dataset_path)
    logger.info(f"Loaded test dataset: {len(test_df)} pairs")
    
    # Initialize evaluator
    evaluator = CrossEncoderEvaluator()
    
    # Prepare training data
    logger.info("Preparing training data...")
    training_data, labels = prepare_training_data(corpus_path, size=500)
    
    models_to_test = []
    
    print("\n🚀 Training and Evaluating Cross-Encoder Models")
    print("=" * 60)
    
    # 1. Random Forest Cross-Encoder
    print("\n📊 1. Random Forest Cross-Encoder")
    print("-" * 40)
    try:
        rf_model = CrossEncoderRanker(model_type='random_forest')
        
        # Convert training data to feature format
        rf_training_pairs, rf_labels = rf_model.create_training_data(
            pd.DataFrame(training_data + [{'text': d['passage']} for d in training_data]).drop_duplicates('text').reset_index(drop=True),
            num_negatives=2
        )
        
        start_time = time.time()
        rf_model.train(rf_training_pairs, rf_labels)
        train_time = time.time() - start_time
        
        models_to_test.append(('Random Forest', rf_model))
        print(f"✅ Trained in {train_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error training Random Forest: {e}")
    
    # 2. XGBoost Cross-Encoder
    print("\n📊 2. XGBoost Cross-Encoder")
    print("-" * 40)
    try:
        xgb_model = XGBoostCrossEncoder(model_type='xgboost')
        
        # Fit feature extractor
        all_texts = [d['query'] for d in training_data] + [d['passage'] for d in training_data]
        xgb_model.feature_extractor.fit(all_texts)
        
        # Convert to features
        xgb_features = []
        xgb_labels = []
        
        for data, label in zip(training_data, labels):
            features = xgb_model.feature_extractor.extract_all_features(data['query'], data['passage'])
            xgb_features.append(features)
            xgb_labels.append(label)
        
        start_time = time.time()
        xgb_model.train(xgb_features, xgb_labels)
        train_time = time.time() - start_time
        
        models_to_test.append(('XGBoost Classifier', xgb_model))
        print(f"✅ Trained in {train_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error training XGBoost: {e}")
    
    # 3. XGBoost LTR - rank:pairwise
    print("\n📊 3. XGBoost LTR (Pairwise)")
    print("-" * 40)
    try:
        # Create LTR training data
        ltr_df = pd.DataFrame([
            {
                'query': d['query'],
                'passage': d['passage'],
                'label': l,
                'difficulty': 'easy' if l == 1 else 'medium',
                'work': d.get('work', 'unknown'),
                'category': d.get('category', 'unknown')
            }
            for d, l in zip(training_data, labels)
        ])
        
        # Save temporary CSV
        temp_ltr_path = "/tmp/ltr_train.csv"
        ltr_df.to_csv(temp_ltr_path, index=False)
        
        ltr_pairwise = XGBoostLTRRanker(
            objective='rank:pairwise',
            n_estimators=100,
            learning_rate=0.3
        )
        
        start_time = time.time()
        ltr_pairwise.train_from_csv(temp_ltr_path)
        train_time = time.time() - start_time
        
        models_to_test.append(('XGBoost LTR (Pairwise)', ltr_pairwise))
        print(f"✅ Trained in {train_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error training XGBoost LTR Pairwise: {e}")
    
    # 4. XGBoost LTR - rank:ndcg
    print("\n📊 4. XGBoost LTR (NDCG)")
    print("-" * 40)
    try:
        ltr_ndcg = XGBoostLTRRanker(
            objective='rank:ndcg',
            n_estimators=100,
            learning_rate=0.3
        )
        
        start_time = time.time()
        ltr_ndcg.train_from_csv(temp_ltr_path)
        train_time = time.time() - start_time
        
        models_to_test.append(('XGBoost LTR (NDCG)', ltr_ndcg))
        print(f"✅ Trained in {train_time:.2f}s")
        
    except Exception as e:
        print(f"❌ Error training XGBoost LTR NDCG: {e}")
    
    # Evaluate all models
    print(f"\n🧪 EVALUATING {len(models_to_test)} MODELS")
    print("=" * 60)
    
    all_results = {}
    
    for model_name, model in models_to_test:
        print(f"\n📊 Evaluating {model_name}...")
        
        try:
            start_time = time.time()
            results = evaluator.evaluate_model(model, test_df, model_name)
            eval_time = time.time() - start_time
            
            all_results[model_name] = results
            
            print(f"✅ Completed in {eval_time:.2f}s")
            print(f"   Accuracy: {results['overall_metrics']['accuracy']:.3f}")
            print(f"   F1: {results['overall_metrics']['f1']:.3f}")
            print(f"   AUC: {results['overall_metrics']['auc']:.3f}")
            print(f"   NDCG@5: {results['ranking_metrics']['ndcg_at_5']:.3f}")
            
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {e}")
    
    # Generate comparison report
    if all_results:
        print(f"\n📊 FINAL COMPARISON REPORT")
        print("=" * 80)
        
        comparison_df = evaluator.compare_models(all_results)
        print(comparison_df.to_string(index=False))
        
        # Save detailed results
        results_path = "/home/sai/Desktop/ullu/data/cross_encoder_evaluation_results.json"
        import json
        with open(results_path, 'w') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for model_name, results in all_results.items():
                json_results[model_name] = {
                    'overall_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                      for k, v in results['overall_metrics'].items()},
                    'ranking_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v 
                                      for k, v in results['ranking_metrics'].items()}
                }
            
            json.dump(json_results, f, indent=2)
        
        print(f"\n💾 Detailed results saved to: {results_path}")
        
        # Find best model
        best_model = comparison_df.loc[comparison_df['F1'].idxmax()]
        print(f"\n🏆 BEST MODEL: {best_model['Model']}")
        print(f"   F1 Score: {best_model['F1']:.3f}")
        print(f"   Accuracy: {best_model['Accuracy']:.3f}")
        print(f"   NDCG@5: {best_model['NDCG@5']:.3f}")
        
        # Analysis by test type
        print(f"\n📈 PERFORMANCE BY TEST TYPE")
        print("-" * 50)
        
        if all_results:
            first_result = list(all_results.values())[0]
            for test_type, metrics in first_result['type_metrics'].items():
                print(f"{test_type}: {metrics['accuracy']:.3f} accuracy ({metrics['count']} pairs)")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 30)
        
        if len(all_results) >= 2:
            models_by_accuracy = comparison_df.sort_values('Accuracy', ascending=False)
            models_by_f1 = comparison_df.sort_values('F1', ascending=False)
            models_by_ndcg = comparison_df.sort_values('NDCG@5', ascending=False)
            
            print(f"Best for Accuracy: {models_by_accuracy.iloc[0]['Model']} ({models_by_accuracy.iloc[0]['Accuracy']:.3f})")
            print(f"Best for F1 Score: {models_by_f1.iloc[0]['Model']} ({models_by_f1.iloc[0]['F1']:.3f})")
            print(f"Best for Ranking: {models_by_ndcg.iloc[0]['Model']} ({models_by_ndcg.iloc[0]['NDCG@5']:.3f})")
            
            # Overall recommendation
            if models_by_f1.iloc[0]['Model'].startswith('XGBoost LTR'):
                print(f"\n🎯 RECOMMENDED: {models_by_f1.iloc[0]['Model']}")
                print("   Reason: Learning-to-Rank is optimal for ranking tasks like quote retrieval")
            else:
                print(f"\n🎯 RECOMMENDED: {models_by_f1.iloc[0]['Model']}")
                print("   Reason: Highest F1 score indicates best balance of precision and recall")
    
    else:
        print("❌ No models were successfully evaluated")

def main():
    """Run comprehensive cross-encoder evaluation."""
    logging.basicConfig(level=logging.INFO)
    
    corpus_path = "/home/sai/Desktop/ullu/data/sanskrit_quotes.csv"
    test_dataset_path = "/home/sai/Desktop/ullu/data/cross_encoder_test_dataset.csv"
    
    if not Path(corpus_path).exists():
        print("❌ Corpus not found. Run data processing first.")
        return
    
    if not Path(test_dataset_path).exists():
        print("❌ Test dataset not found. Run cross_encoder_evaluation.py first.")
        return
    
    print("🧪 COMPREHENSIVE CROSS-ENCODER EVALUATION")
    print("Testing 4 different models on high-quality test dataset")
    print("Models: Random Forest, XGBoost Classifier, XGBoost LTR (Pairwise), XGBoost LTR (NDCG)")
    
    train_and_evaluate_models(test_dataset_path, corpus_path)

if __name__ == "__main__":
    main()