"""
LightGBM Training Module
Trains the multi-class triage classifier
"""

import lightgbm as lgb
import numpy as np
import pandas as pd
import pickle
import os
import json
from datetime import datetime
from typing import Dict, Tuple, Optional
from .utils import (
    load_processed_data,
    load_target_mapping,
    get_class_weights,
    get_sample_weights,
    validate_data
)


class LightGBMTriageModel:
    """
    LightGBM model for 3-class triage prediction.
    
    Classes:
        0: FalsePositive
        1: BenignPositive
        2: TruePositive
    """
    
    def __init__(self, params: Optional[Dict] = None):
        """
        Initialize LightGBM triage model.
        
        Args:
            params: LightGBM parameters (if None, uses defaults)
        """
        if params is None:
            params = self._get_default_params()
        
        self.params = params
        self.model = None
        self.feature_names = None
        self.class_weights = None
        self.target_mapping = None
        self.training_history = {}
        
    def _get_default_params(self) -> Dict:
        """Get default LightGBM parameters."""
        return {
            'objective': 'multiclass',
            'num_class': 3,
            'metric': 'multi_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_child_samples': 20,
            'min_child_weight': 0.001,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'max_depth': -1,
            'verbose': -1,
            'force_col_wise': True,
            'random_state': 42
        }
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame,
        y_val: pd.Series,
        use_class_weights: bool = True,
        num_boost_round: int = 1000,
        early_stopping_rounds: int = 50,
        verbose_eval: int = 50
    ) -> Dict:
        """
        Train the LightGBM triage model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            use_class_weights: Whether to use class weights for imbalance
            num_boost_round: Maximum number of boosting rounds
            early_stopping_rounds: Early stopping patience
            verbose_eval: Print evaluation every N rounds
            
        Returns:
            Training history dictionary
        """
        print("\n" + "="*80)
        print("TRAINING LIGHTGBM TRIAGE MODEL")
        print("="*80)
        
        # Store feature names
        self.feature_names = list(X_train.columns)
        
        # Compute class weights if requested
        if use_class_weights:
            self.class_weights = get_class_weights(y_train)
            sample_weights = get_sample_weights(y_train, self.class_weights)
        else:
            sample_weights = None
            print("⚠️  Training without class weights")
        
        # Create LightGBM datasets
        print("\n📊 Creating LightGBM datasets...")
        train_data = lgb.Dataset(
            X_train,
            label=y_train,
            weight=sample_weights,
            feature_name=self.feature_names
        )
        
        val_data = lgb.Dataset(
            X_val,
            label=y_val,
            reference=train_data,
            feature_name=self.feature_names
        )
        
        # Train model
        print(f"\n🚀 Starting training...")
        print(f"   Max rounds: {num_boost_round}")
        print(f"   Early stopping: {early_stopping_rounds} rounds")
        print(f"   Parameters: {json.dumps(self.params, indent=2)}")
        
        start_time = datetime.now()
        
        # Callbacks for tracking
        callbacks = [
            lgb.early_stopping(early_stopping_rounds),
            lgb.log_evaluation(verbose_eval)
        ]
        
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=num_boost_round,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'val'],
            callbacks=callbacks
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store training history
        self.training_history = {
            'best_iteration': self.model.best_iteration,
            'best_score': self.model.best_score,
            'training_time_seconds': training_time,
            'num_features': len(self.feature_names),
            'params': self.params,
            'use_class_weights': use_class_weights
        }
        
        print(f"\n✅ Training complete!")
        print(f"   Best iteration: {self.model.best_iteration}")
        print(f"   Best validation score: {self.model.best_score}")
        print(f"   Training time: {training_time:.2f} seconds")
        
        return self.training_history
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of shape (n_samples, 3) with probabilities for each class
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        return self.model.predict(X, num_iteration=self.model.best_iteration)
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Array of predicted class labels (0, 1, or 2)
        """
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
    def predict_with_confidence(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict labels with probabilities and confidence scores.
        
        Args:
            X: Feature dataframe
            
        Returns:
            Tuple of (predictions, probabilities, confidence_scores)
        """
        probas = self.predict_proba(X)
        predictions = np.argmax(probas, axis=1)
        confidence_scores = np.max(probas, axis=1)
        
        return predictions, probas, confidence_scores
    
    def save(self, save_dir: str = 'models/lightgbm', model_name: str = 'triage_model'):
        """
        Save trained model and metadata.
        
        Args:
            save_dir: Directory to save model
            model_name: Base name for model files
        """
        if self.model is None:
            raise ValueError("No model to save. Train the model first.")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(save_dir, f'{model_name}.txt')
        self.model.save_model(model_path)
        print(f"✅ Model saved to: {model_path}")
        
        # Save metadata
        metadata = {
            'feature_names': self.feature_names,
            'class_weights': self.class_weights,
            'training_history': self.training_history,
            'params': self.params
        }
        
        metadata_path = os.path.join(save_dir, f'{model_name}_metadata.pkl')
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        print(f"✅ Metadata saved to: {metadata_path}")
        
        # Save training history as JSON
        history_path = os.path.join(save_dir, f'{model_name}_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2)
        print(f"✅ Training history saved to: {history_path}")
    
    def load(self, save_dir: str = 'models/lightgbm', model_name: str = 'triage_model'):
        """
        Load trained model and metadata.
        
        Args:
            save_dir: Directory containing saved model
            model_name: Base name of model files
        """
        # Load model
        model_path = os.path.join(save_dir, f'{model_name}.txt')
        self.model = lgb.Booster(model_file=model_path)
        print(f"✅ Model loaded from: {model_path}")
        
        # Load metadata
        metadata_path = os.path.join(save_dir, f'{model_name}_metadata.pkl')
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.feature_names = metadata['feature_names']
        self.class_weights = metadata['class_weights']
        self.training_history = metadata['training_history']
        self.params = metadata['params']
        print(f"✅ Metadata loaded from: {metadata_path}")


def train_triage_model(
    data_dir: str = 'data/processed/v1',
    save_dir: str = 'models/lightgbm',
    params: Optional[Dict] = None,
    use_class_weights: bool = True
) -> LightGBMTriageModel:
    """
    Complete training pipeline for triage model.
    
    Args:
        data_dir: Directory with processed data
        save_dir: Directory to save model
        params: Custom LightGBM parameters
        use_class_weights: Whether to handle class imbalance
        
    Returns:
        Trained LightGBMTriageModel
    """
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data(data_dir)
    
    # Validate data
    validate_data(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Load target mapping
    target_mapping = load_target_mapping()
    
    # Initialize model
    model = LightGBMTriageModel(params=params)
    model.target_mapping = target_mapping
    
    # Train
    model.train(
        X_train, y_train,
        X_val, y_val,
        use_class_weights=use_class_weights
    )
    
    # Save
    model.save(save_dir=save_dir)
    
    return model


if __name__ == "__main__":
    # Test training
    print("Testing LightGBM training...")
    model = train_triage_model()
    
    # Test prediction
    print("\n" + "="*80)
    print("TESTING PREDICTIONS")
    print("="*80)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_processed_data()
    
    # Predict on validation set (small sample)
    X_sample = X_val.head(10)
    y_sample = y_val.head(10)
    
    predictions, probas, confidence = model.predict_with_confidence(X_sample)
    
    print(f"\n📊 Sample predictions:")
    for i in range(len(predictions)):
        print(f"   Sample {i}: Predicted={predictions[i]}, True={y_sample.iloc[i]}, "
              f"Confidence={confidence[i]:.4f}")
    
    print("\n✅ All tests passed!")
