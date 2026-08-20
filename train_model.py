"""
HeartGuard AI — Model Training Script
======================================
Trains the 6-model ensemble and saves it to model.pkl
Run this script to pre-train the model for faster app startup.

Usage:
    python train_model.py
"""
import os
import sys
import joblib

# Add backend directory to Python path
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, 'backend'))

from ml_core import train_all_models

def main():
    print("=" * 60)
    print("  HeartGuard AI — Training ML Models")
    print("=" * 60)

    # Train all models
    ensemble_model, individual_models, scaler, cv_scores = train_all_models()

    # Save the trained model
    model_data = {
        'ensemble_model': ensemble_model,
        'individual_models': individual_models,
        'scaler': scaler,
        'cv_scores': cv_scores,
    }

    model_path = os.path.join(PROJECT_DIR, 'model.pkl')
    joblib.dump(model_data, model_path)

    print(f"\n[OK] Model saved to {model_path}")
    print(f"     Ensemble CV Accuracy: {cv_scores['Ensemble (Soft Vote)']['mean']:.4f} ± {cv_scores['Ensemble (Soft Vote)']['std']:.4f}")
    print("=" * 60)

if __name__ == '__main__':
    main()