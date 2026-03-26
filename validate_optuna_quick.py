#!/usr/bin/env python
"""
Quick validation of Optuna TabNet tuner
Tests initialization with minimal data without full training
"""

import sys
import numpy as np
from pathlib import Path

print("\n" + "=" * 70)
print("OPTUNA TABNET TUNER - QUICK VALIDATION")
print("=" * 70)

print("\n[STEP 1] Checking imports...")
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    from pytorch_tabnet.tab_model import TabNetClassifier
    from sklearn.metrics import f1_score
    print("  ✓ All imports successful")
except ImportError as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

print("\n[STEP 2] Testing TabNetTuner class...")
try:
    sys.path.insert(0, str(Path.cwd()))
    from src.tuning.optuna_tabnet import TabNetTuner
    
    # Create minimal synthetic data for testing
    n_samples_train = 1000
    n_samples_val = 200
    n_features = 44
    n_classes = 3
    
    X_train = np.random.randn(n_samples_train, n_features).astype(np.float32)
    X_val = np.random.randn(n_samples_val, n_features).astype(np.float32)
    y_train = np.random.randint(0, n_classes, n_samples_train)
    y_val = np.random.randint(0, n_classes, n_samples_val)
    
    # Compute class weights
    from src.models.tabnet.utils import compute_tabnet_class_weights
    class_weights = compute_tabnet_class_weights(y_train)
    
    # Create tuner instance
    tuner = TabNetTuner(
        X_train, X_val, y_train, y_val,
        class_weights,
        n_trials=2,  # Just 2 trials for validation
        pruning_enabled=True,
        verbose=False,
    )
    
    print("  ✓ TabNetTuner instantiation successful")
    print(f"      Search space size: {tuner.search_space_size}")
    print(f"      Max trials: {tuner.n_trials}")
    
except Exception as e:
    print(f"  ✗ TabNetTuner initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n[STEP 3] Testing single trial...")
try:
    study = optuna.create_study(
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(),
        direction='maximize',
    )
    
    print("  Running trial...")
    study.optimize(tuner.objective, n_trials=1, show_progress_bar=False)
    
    best_trial = study.best_trial
    best_f1 = best_trial.value
    
    print(f"  ✓ Trial completed")
    print(f"      Best F1: {best_f1:.4f}")
    print(f"      Parameters tested:")
    for key, val in list(best_trial.params.items())[:3]:
        print(f"          {key}: {val}")
    
except Exception as e:
    print(f"  ✗ Trial failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 70)
print("✓ VALIDATION SUCCESSFUL")
print("=" * 70)

print("\n[NEXT STEPS]")
print("""
The Optuna TabNet tuner is ready to use!

To run full hyperparameter tuning on actual data:
  
  python src/tuning/optuna_tabnet.py
  
  This will:
  1. Load SOC alert data from data/processed/v1/
  2. Scale features using QuantileTransformer
  3. Compute class weights for imbalance
  4. Run 30 trials of hyperparameter optimization
  5. Save best parameters to models/tuning/optuna_results.json
  
Expected runtime: ~2-4 hours on CPU (parallelizable on GPU)
Objective: Maximize macro-F1 score on validation set
""")

print("✓ Ready for production execution!")
