#!/usr/bin/env python3
"""
Quick start script to train models and see results.
Run with: python quick_start.py [model]

Options:
  python quick_start.py               # Train all models
  python quick_start.py svm           # Train improved SVM only
  python quick_start.py ensemble      # Train ensemble only
  python quick_start.py test          # Test with sample data
"""

import sys
import os

def main():
    # Ensure data is preprocessed
    print("Checking if data is preprocessed...")
    if not os.path.exists("data/processed/train.csv"):
        print("Preprocessing data...")
        from src.preprocess import save_data
        save_data()
    
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()
    else:
        model_type = "all"
    
    if model_type in ["all", "svm"]:
        print("\n" + "="*70)
        print("Training Improved SVM Model...")
        print("="*70)
        try:
            from src.train_improved_svm import train_improved_svm
            train_improved_svm()
        except Exception as e:
            print(f"Error: {e}")
    
    if model_type in ["all", "ensemble"]:
        print("\n" + "="*70)
        print("Training Ensemble Model...")
        print("="*70)
        try:
            from src.train_ensemble import train_ensemble
            train_ensemble()
        except Exception as e:
            print(f"Error: {e}")
    
    if model_type == "test":
        print("\n" + "="*70)
        print("Testing Predictions...")
        print("="*70)
        from src.predict_improved import predict_news_details
        
        test_samples = [
            "Breaking: Government Launches New Education Initiative",
            "SHOCKING!!! You won't BELIEVE what happened NEXT!!!",
            "Scientists discover new renewable energy source",
            "EXCLUSIVE SCANDAL EXPOSED - See what the media won't tell you!"
        ]
        
        for text in test_samples:
            result = predict_news_details(text)
            print(f"\nText: {text[:60]}...")
            print(f"Prediction: {result['label']}")
            print(f"Confidence: {result['confidence']}%")
            print(f"Real: {result['real_percentage']}%, Fake: {result['fake_percentage']}%")
    
    print("\n" + "="*70)
    print("DONE!")
    print("="*70)


if __name__ == "__main__":
    main()
