"""
TabNet model training script - Data loading and validation.

Loads processed v1 data and performs validation checks before model training.
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import Tuple, Dict, Any


# Data paths
DATA_DIR = "data/processed/v1"


def load_tabnet_data() -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Load TabNet training data from processed v1 directory.
    
    Returns
    -------
    Tuple of numpy arrays:
        X_train, X_val, X_test, y_train, y_val, y_test
    
    Raises
    ------
    FileNotFoundError
        If any required data file is missing
    ValueError
        If data validation fails
    """
    print("=" * 60)
    print("Loading TabNet Data")
    print("=" * 60)
    
    # Construct full file paths
    paths = {
        'X_train': os.path.join(DATA_DIR, 'X_train.csv'),
        'X_val': os.path.join(DATA_DIR, 'X_val.csv'),
        'X_test': os.path.join(DATA_DIR, 'X_test.csv'),
        'y_train': os.path.join(DATA_DIR, 'y_train.csv'),
        'y_val': os.path.join(DATA_DIR, 'y_val.csv'),
        'y_test': os.path.join(DATA_DIR, 'y_test.csv'),
    }
    
    # Verify all files exist
    print("\n[LOAD] Checking file existence...")
    for name, path in paths.items():
        if not os.path.exists(path):
            raise FileNotFoundError(f"Required data file not found: {path}")
        print(f"  ✓ {name}: {path}")
    
    # Load data
    print("\n[LOAD] Loading data files...")
    X_train = pd.read_csv(paths['X_train'])
    X_val = pd.read_csv(paths['X_val'])
    X_test = pd.read_csv(paths['X_test'])
    y_train = pd.read_csv(paths['y_train']).iloc[:, 0]  # Extract first column as Series
    y_val = pd.read_csv(paths['y_val']).iloc[:, 0]
    y_test = pd.read_csv(paths['y_test']).iloc[:, 0]
    
    print("  ✓ Data loaded successfully")
    
    # Validate and display shape information
    print("\n[VALIDATION] Shape Information:")
    print(f"  X_train: {X_train.shape}")
    print(f"  X_val:   {X_val.shape}")
    print(f"  X_test:  {X_test.shape}")
    print(f"  y_train: {y_train.shape}")
    print(f"  y_val:   {y_val.shape}")
    print(f"  y_test:  {y_test.shape}")
    
    # Validate consistency of number of features
    n_features_train = X_train.shape[1]
    if X_val.shape[1] != n_features_train:
        raise ValueError(
            f"Feature mismatch: X_train has {n_features_train} features "
            f"but X_val has {X_val.shape[1]}"
        )
    if X_test.shape[1] != n_features_train:
        raise ValueError(
            f"Feature mismatch: X_train has {n_features_train} features "
            f"but X_test has {X_test.shape[1]}"
        )
    print("  ✓ Feature count consistent across splits")
    
    # Validate sample count consistency with labels
    if X_train.shape[0] != len(y_train):
        raise ValueError(
            f"Sample count mismatch: X_train has {X_train.shape[0]} samples "
            f"but y_train has {len(y_train)}"
        )
    if X_val.shape[0] != len(y_val):
        raise ValueError(
            f"Sample count mismatch: X_val has {X_val.shape[0]} samples "
            f"but y_val has {len(y_val)}"
        )
    if X_test.shape[0] != len(y_test):
        raise ValueError(
            f"Sample count mismatch: X_test has {X_test.shape[0]} samples "
            f"but y_test has {len(y_test)}"
        )
    print("  ✓ Sample count consistent between features and targets")
    
    # Check for missing values
    print("\n[VALIDATION] Missing Values Check:")
    missing_x_train = X_train.isnull().sum().sum()
    missing_x_val = X_val.isnull().sum().sum()
    missing_x_test = X_test.isnull().sum().sum()
    missing_y_train = y_train.isnull().sum()
    missing_y_val = y_val.isnull().sum()
    missing_y_test = y_test.isnull().sum()
    
    print(f"  X_train missing values: {missing_x_train}")
    print(f"  X_val missing values:   {missing_x_val}")
    print(f"  X_test missing values:  {missing_x_test}")
    print(f"  y_train missing values: {missing_y_train}")
    print(f"  y_val missing values:   {missing_y_val}")
    print(f"  y_test missing values:  {missing_y_test}")
    
    total_missing = (missing_x_train + missing_x_val + missing_x_test + 
                     missing_y_train + missing_y_val + missing_y_test)
    if total_missing > 0:
        raise ValueError(f"Missing values detected: total={total_missing}")
    print("  ✓ No missing values detected")
    
    # Verify all features are numeric
    print("\n[VALIDATION] Feature Type Check:")
    
    # Check X_train
    non_numeric_cols_train = X_train.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    if non_numeric_cols_train:
        raise ValueError(
            f"X_train contains non-numeric columns: {non_numeric_cols_train}"
        )
    
    # Check X_val
    non_numeric_cols_val = X_val.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    if non_numeric_cols_val:
        raise ValueError(
            f"X_val contains non-numeric columns: {non_numeric_cols_val}"
        )
    
    # Check X_test
    non_numeric_cols_test = X_test.select_dtypes(
        exclude=[np.number]
    ).columns.tolist()
    if non_numeric_cols_test:
        raise ValueError(
            f"X_test contains non-numeric columns: {non_numeric_cols_test}"
        )
    
    print(f"  X_train features: {n_features_train} (all numeric)")
    print(f"  X_val features: {X_val.shape[1]} (all numeric)")
    print(f"  X_test features: {X_test.shape[1]} (all numeric)")
    print(f"  Target variable types: {set(y_train.dtype for y_train in [y_train, y_val, y_test])}")
    print("  ✓ All feature columns are numeric")
    
    # Display summary statistics
    print("\n[SUMMARY] Data Statistics:")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Validation samples: {X_val.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}")
    print(f"  Total samples: {X_train.shape[0] + X_val.shape[0] + X_test.shape[0]}")
    print(f"  Features: {n_features_train}")
    print(f"  Target classes: {len(np.unique(y_train))}")
    print(f"  Class distribution in training set:")
    for class_label, count in sorted(zip(*np.unique(y_train, return_counts=True))):
        pct = 100 * count / len(y_train)
        print(f"    Class {class_label}: {count} samples ({pct:.1f}%)")
    
    print("\n" + "=" * 60)
    print("✓ Data loading and validation successful!")
    print("=" * 60)
    
    # Convert to numpy arrays
    return (
        X_train.values,
        X_val.values,
        X_test.values,
        y_train.values,
        y_val.values,
        y_test.values
    )


def get_data_info() -> Dict[str, Any]:
    """
    Get information about loaded data without full validation.
    
    Returns
    -------
    Dict containing:
        - shapes: Dictionary of shapes for each split
        - features: Number of features
        - classes: Number of unique classes in training data
    """
    paths = {
        'X_train': os.path.join(DATA_DIR, 'X_train.csv'),
        'X_val': os.path.join(DATA_DIR, 'X_val.csv'),
        'X_test': os.path.join(DATA_DIR, 'X_test.csv'),
        'y_train': os.path.join(DATA_DIR, 'y_train.csv'),
    }
    
    X_train = pd.read_csv(paths['X_train'])
    y_train = pd.read_csv(paths['y_train']).iloc[:, 0]
    
    return {
        'shapes': {
            'X_train': X_train.shape,
            'y_train': y_train.shape,
        },
        'features': X_train.shape[1],
        'classes': len(np.unique(y_train)),
    }


if __name__ == "__main__":
    """Main execution - load and validate data"""
    try:
        X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()
    except Exception as e:
        print(f"\n[ERROR] {str(e)}")
        raise
