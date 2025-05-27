#!/usr/bin/env python3
"""
Simple Cross-Encoder Testing Script for ULLU
Tests different cross-encoder models on quality test data.
"""

import sys
sys.path.append('src')

import pandas as pd
import numpy as np
from pathlib import Path
import time

def test_basic_cross_encoder():
    """Test the basic Random Forest cross-encoder."""
    from ranking.cross_encoder import CrossEncoderRanker
    
    print("🌲 Testing Random Forest Cross-Encoder")
    print("-" * 40)
    
    # Load test data
    test_path = "data/cross_encoder_test_dataset.csv"
    if not Path(test_path).exists():
        print("❌ Test dataset not found. Run evaluation script first.")
        return None
    
    test_df = pd.read_csv(test_path)
    print(f"📊 Test dataset: {len(test_df)} pairs")
    
    # Create simple training data  
    quotes_df = pd.read_csv("data/sanskrit_quotes.csv")
    
    # Initialize and train model
    model = CrossEncoderRanker(model_type='random_forest')
    
    print("🔧 Training model...")
    start_time = time.time()
    
    training_pairs, labels = model.create_training_data(quotes_df.head(100), num_negatives=2)
    model.train(training_pairs, labels)
    
    train_time = time.time() - start_time
    print(f"✅ Trained in {train_time:.2f}s")
    
    # Test on sample queries
    test_queries = [
        {
            'query': 'dharma viṣṇu',
            'candidates': [
                {'text': 'dharmaviṣṇupriya śrīkṛṣṇa paramātma sanātana', 'id': 1},
                {'text': 'rājya dhana āryāṇām viṣṇave namaḥ', 'id': 2},
                {'text': 'gaṅgā yamunā sarasvatī devī', 'id': 3}
            ]
        },
        {
            'query': 'namaḥ śivāya',
            'candidates': [
                {'text': 'namaḥ śivāya oṃ maheśvarāya', 'id': 4},
                {'text': 'kṛṣṇa rādhā prema līlā', 'id': 5},
                {'text': 'gaṅgā gaṅgeti yo brūyāt', 'id': 6}
            ]
        }
    ]
    
    print(f"\n📊 Testing Rankings:")
    
    for i, test_case in enumerate(test_queries, 1):
        query = test_case['query']
        candidates = test_case['candidates']
        
        print(f"\n{i}. Query: '{query}'")
        
        ranked = model.rank_passages(query, candidates)
        
        for rank, result in enumerate(ranked, 1):
            score = result.get('ranking_score', 0)
            print(f"   Rank {rank}: Score={score:.3f} | {result['text'][:50]}...")
    
    return model

def simple_evaluation_metrics(test_df, model):
    """Simple evaluation on exact match test cases."""
    print(f"\n📈 Simple Evaluation Metrics")
    print("-" * 30)
    
    # Test only exact matches (should score highest)
    exact_matches = test_df[test_df['test_type'] == 'exact_match'].head(50)
    
    if len(exact_matches) == 0:
        print("❌ No exact match test cases found")
        return
    
    correct_rankings = 0
    total_tests = 0
    
    for _, row in exact_matches.iterrows():
        query = row['query']
        correct_passage = row['passage']
        
        # Create fake candidates (correct + random)
        random_passages = test_df[test_df['test_type'] == 'negative_random'].sample(2)
        
        candidates = [
            {'text': correct_passage, 'id': 'correct'},
            {'text': random_passages.iloc[0]['passage'], 'id': 'random1'},
            {'text': random_passages.iloc[1]['passage'], 'id': 'random2'}
        ]
        
        # Rank candidates
        ranked = model.rank_passages(query, candidates)
        
        # Check if correct passage ranked first
        if ranked[0]['id'] == 'correct':
            correct_rankings += 1
        
        total_tests += 1
    
    accuracy = correct_rankings / total_tests if total_tests > 0 else 0
    
    print(f"Exact Match Accuracy: {accuracy:.1%} ({correct_rankings}/{total_tests})")
    print(f"This measures if the model ranks exact substring matches highest")
    
    return accuracy

def show_test_dataset_stats():
    """Show statistics about the test dataset."""
    test_path = "data/cross_encoder_test_dataset.csv"
    
    if not Path(test_path).exists():
        print("❌ Test dataset not found")
        return
    
    df = pd.read_csv(test_path)
    
    print(f"\n📊 Test Dataset Statistics")
    print("=" * 40)
    print(f"Total test pairs: {len(df):,}")
    print(f"Positive pairs: {len(df[df['label'] == 1]):,}")
    print(f"Negative pairs: {len(df[df['label'] == 0]):,}")
    
    print(f"\nTest Types:")
    for test_type, count in df['test_type'].value_counts().items():
        print(f"  {test_type}: {count:,}")
    
    print(f"\nDifficulty Levels:")
    for difficulty, count in df['difficulty'].value_counts().items():
        print(f"  {difficulty}: {count:,}")
    
    print(f"\nExpected Performance Guide:")
    print(f"  exact_match: Should achieve >95% accuracy")
    print(f"  partial_match: Should achieve >70% accuracy") 
    print(f"  semantic_match: Should achieve >60% accuracy")
    print(f"  negative_*: Should achieve >90% accuracy (rejecting)")

def main():
    """Run simple cross-encoder tests."""
    print("🧪 ULLU Cross-Encoder Quality Testing")
    print("Testing cross-encoder models on Sanskrit quote ranking")
    print("=" * 60)
    
    # Check if data exists
    if not Path("data/sanskrit_quotes.csv").exists():
        print("❌ Sanskrit quotes dataset not found!")
        print("   Run: python3 src/data_processing.py")
        return
    
    if not Path("data/cross_encoder_test_dataset.csv").exists():
        print("❌ Test dataset not found!")
        print("   Run: python3 src/evaluation/cross_encoder_evaluation.py")
        return
    
    # Show test dataset info
    show_test_dataset_stats()
    
    # Test basic model
    model = test_basic_cross_encoder()
    
    if model:
        # Run simple evaluation
        test_df = pd.read_csv("data/cross_encoder_test_dataset.csv")
        accuracy = simple_evaluation_metrics(test_df, model)
        
        print(f"\n🎯 Results Summary:")
        print(f"✅ Model trained successfully")
        print(f"✅ Exact match accuracy: {accuracy:.1%}")
        
        if accuracy > 0.8:
            print(f"🏆 EXCELLENT: Model performs very well on exact matches")
        elif accuracy > 0.6:
            print(f"✅ GOOD: Model performs reasonably well")
        else:
            print(f"⚠️  NEEDS IMPROVEMENT: Consider more training data or features")
        
        print(f"\n💡 Next Steps:")
        print(f"   1. Generate larger training dataset: python3 src/data_generation.py")
        print(f"   2. Try XGBoost LTR: python3 src/ranking/xgboost_ltr.py")
        print(f"   3. Run full evaluation: python3 src/evaluation/run_cross_encoder_tests.py")

if __name__ == "__main__":
    main()