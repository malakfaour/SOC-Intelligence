#!/usr/bin/env python
"""Quick test of model saving functionality"""

import sys
import os
sys.path.insert(0, 'src/training')
sys.path.insert(0, 'src/models/tabnet')

from train_tabnet import load_tabnet_data
from utils import (
    scale_tabnet_features,
    compute_tabnet_class_weights,
    save_tabnet_model,
    load_tabnet_model,
    TabNetScaler,
)

print("\nTesting TabNet Model Save/Load Functionality\n")

try:
    # Load data
    print("[STEP 1] Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()
    print("✓ Data loaded")
    
    # Scale features
    print("\n[STEP 2] Scaling features...")
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_tabnet_features(
        X_train, X_val, X_test, verbose=False
    )
    print("✓ Features scaled")
    
    # Compute class weights
    print("\n[STEP 3] Computing class weights...")
    class_weights = compute_tabnet_class_weights(y_train, verbose=False)
    print(f"✓ Class weights computed: {class_weights}")
    
    # Create mock model (we can't train without torch/pytorch-tabnet working)
    # But we can test the save/load infrastructure
    print("\n[STEP 4] Testing save/load infrastructure...")
    
    class MockModel:
        """Mock model for testing save/load without TabNet"""
        def __init__(self):
            self.name = "mock_model"
    
    mock_model = MockModel()
    
    # Test save function with mock model
    print("\n[STEP 5] Testing save_tabnet_model function...")
    try:
        artifact_paths = save_tabnet_model(
            model=mock_model,
            scaler=scaler,
            class_weights=class_weights,
            model_dir="models/tabnet_test",
            model_name="test_model",
            verbose=True
        )
        print("✓ Save function executed successfully")
    except Exception as e:
        print(f"✗ Save function error: {e}")
        raise
    
    # Test load function
    print("\n[STEP 6] Testing load_tabnet_model function...")
    try:
        loaded_model, loaded_scaler, loaded_config = load_tabnet_model(
            model_dir="models/tabnet_test",
            model_name="test_model",
            verbose=True
        )
        print("✓ Load function executed successfully")
    except Exception as e:
        print(f"✗ Load function error: {e}")
        raise
    
    # Cleanup test directory
    import shutil
    if os.path.exists("models/tabnet_test"):
        shutil.rmtree("models/tabnet_test")
        print("\n✓ Cleanup complete")
    
    print("\n" + "=" * 60)
    print("✓ All save/load tests passed!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n[ERROR] {str(e)}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
