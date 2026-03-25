"""
<<<<<<< HEAD
Data cleaning module for GUIDE dataset.

Handles missing values, irrelevant column removal, and data validation.
"""

import pandas as pd
import numpy as np
from typing import List, Tuple


# Columns to drop as they are identifiers or not useful for prediction
IDENTIFIER_COLUMNS = [
    'Id',
    'AlertId',
    'HashId',
    'UUID',
    'RecordId',
    'Timestamp',
    'Date',
    'Time',
    'SessionId',
    'EventId'
]

# Target column
TARGET_COLUMN = 'IncidentGrade'


def identify_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str], str]:
    """
    Identify numerical, categorical, and target columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    Tuple[List[str], List[str], str]
        (numerical_cols, categorical_cols, target_col)
    """
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Remove target from feature columns if present
    if TARGET_COLUMN in numerical_cols:
        numerical_cols.remove(TARGET_COLUMN)
    if TARGET_COLUMN in categorical_cols:
        categorical_cols.remove(TARGET_COLUMN)
    
    return numerical_cols, categorical_cols, TARGET_COLUMN


def drop_irrelevant_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop identifier and irrelevant columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    
    Returns
    -------
    pd.DataFrame
        DataFrame with irrelevant columns removed
    """
    cols_to_drop = [col for col in IDENTIFIER_COLUMNS if col in df.columns]
    
    if cols_to_drop:
        print(f"✓ Dropping {len(cols_to_drop)} identifier columns: {cols_to_drop}")
        df = df.drop(columns=cols_to_drop)
    
    return df


def handle_missing_values(
    df: pd.DataFrame,
    numerical_cols: List[str],
    categorical_cols: List[str],
    numerical_strategy: str = 'median',
    categorical_fill_value: str = 'unknown'
) -> pd.DataFrame:
    """
    Fill missing values for numerical and categorical columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame
    numerical_cols : List[str]
        List of numerical column names
    categorical_cols : List[str]
        List of categorical column names
    numerical_strategy : str
        Strategy for filling numerical missing values ('mean' or 'median')
    categorical_fill_value : str
        Value to fill categorical missing values
    
    Returns
    -------
    pd.DataFrame
        DataFrame with missing values handled
    """
    missing_info = df.isnull().sum()
    if missing_info.sum() > 0:
        print(f"✓ Found {missing_info.sum()} missing values")
        print(f"  {missing_info[missing_info > 0].to_dict()}")
    
    # Fill numerical columns
    for col in numerical_cols:
        if col in df.columns and df[col].isnull().any():
            if numerical_strategy == 'mean':
                fill_value = df[col].mean()
            else:  # median
                fill_value = df[col].median()
            df[col].fillna(fill_value, inplace=True)
            print(f"  → Filled {col} (numeric) with {numerical_strategy}: {fill_value:.2f}")
    
    # Fill categorical columns
    for col in categorical_cols:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(categorical_fill_value, inplace=True)
            print(f"  → Filled {col} (categorical) with '{categorical_fill_value}'")
    
    return df


def clean_data(
    df: pd.DataFrame,
    drop_identifiers: bool = True,
    numerical_strategy: str = 'median',
    categorical_fill_value: str = 'unknown'
) -> pd.DataFrame:
    """
    Main cleaning pipeline.
    
    Applies all cleaning steps in sequence:
    1. Drop irrelevant/identifier columns
    2. Handle missing values
    
    Parameters
    ----------
    df : pd.DataFrame
        Raw input DataFrame
    drop_identifiers : bool
        Whether to drop identifier columns
    numerical_strategy : str
        Strategy for filling numerical missing values
    categorical_fill_value : str
        Value for filling categorical missing values
    
    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame
    """
    print("\n=== CLEANING DATA ===")
    
    # Step 1: Drop identifier columns
    if drop_identifiers:
        df = drop_irrelevant_columns(df)
    
    # Step 2: Identify column types
    numerical_cols, categorical_cols, target_col = identify_column_types(df)
    print(f"\nColumn types identified:")
    print(f"  - Numerical: {len(numerical_cols)}")
    print(f"  - Categorical: {len(categorical_cols)}")
    print(f"  - Target: {target_col}")
    
    # Step 3: Handle missing values
    df = handle_missing_values(
        df,
        numerical_cols,
        categorical_cols,
        numerical_strategy=numerical_strategy,
        categorical_fill_value=categorical_fill_value
    )
    
    print(f"\n✓ Cleaning complete. Final shape: {df.shape}")
    return df
=======
Minimal data cleaning module.
- Fill numerical missing values with median
- Fill categorical missing values with 'unknown'
- NO encoding, NO scaling
"""
import pandas as pd
import numpy as np


def clean_data(df, target_col='IncidentGrade', verbose=True):
    """
    Minimal data cleaning.
    
    Args:
        df: Input DataFrame
        target_col: Name of target column (will not be cleaned)
        verbose: Print debug information
    
    Returns:
        Cleaned DataFrame
    """
    if verbose:
        print(f"\n[CLEANING] Starting data cleaning...")
        print(f"[CLEANING] Input shape: {df.shape}")
        print(f"[CLEANING] Missing values before cleaning:")
        missing_before = df.isnull().sum()
        print(f"  Total NaN cells: {missing_before.sum()}")
        print(f"  Columns with missing: {(missing_before > 0).sum()}")
    
    df_clean = df.copy()
    
    # Separate target and features
    if target_col in df_clean.columns:
        target = df_clean[target_col]
        features = df_clean.drop(columns=[target_col])
    else:
        features = df_clean
        target = None
    
    # Identify numeric and categorical columns
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features.select_dtypes(include=['object']).columns.tolist()
    
    if verbose:
        print(f"\n[CLEANING] Column types:")
        print(f"  Numeric columns: {len(numeric_cols)}")
        print(f"  Categorical columns: {len(categorical_cols)}")
    
    # Fill numeric missing values with median
    for col in numeric_cols:
        if features[col].isnull().sum() > 0:
            median_val = features[col].median()
            if verbose:
                print(f"[CLEANING] Filling {col} with median: {median_val}")
            features[col] = features[col].fillna(median_val)
    
    # Fill categorical missing values with 'unknown'
    for col in categorical_cols:
        if features[col].isnull().sum() > 0:
            if verbose:
                print(f"[CLEANING] Filling {col} with 'unknown'")
            features[col] = features[col].fillna('unknown')
    
    # Handle target column missing values
    if target is not None:
        target_missing = target.isnull().sum()
        if target_missing > 0:
            if verbose:
                print(f"\n[CLEANING] Target column has {target_missing} missing values")
                print(f"[CLEANING] Removing rows with missing target...")
            # Remove rows where target is NaN
            valid_idx = target.notna()
            features = features[valid_idx]
            target = target[valid_idx]
            if verbose:
                print(f"[CLEANING] New shape: {features.shape}")
    
    # Reconstruct dataframe
    if target is not None:
        df_clean = features.copy()
        df_clean[target_col] = target
    else:
        df_clean = features
    
    if verbose:
        print(f"\n[CLEANING] Output shape: {df_clean.shape}")
        print(f"[CLEANING] Missing values after cleaning:")
        missing_after = df_clean.isnull().sum()
        print(f"  Total NaN cells: {missing_after.sum()}")
        if missing_after.sum() == 0:
            print(f"  ✓ All missing values handled!")
        else:
            print(f"  ✗ Still missing values in: {missing_after[missing_after > 0].index.tolist()}")
    
    return df_clean


if __name__ == "__main__":
    # Test
    df = pd.DataFrame({
        'num1': [1, 2, np.nan, 4, 5],
        'num2': [1.5, np.nan, 3.5, 4.5, 5.5],
        'cat1': ['a', 'b', 'a', np.nan, 'c'],
        'target': ['TP', 'BP', 'FP', 'TP', 'BP']
    })
    print("Input:\n", df)
    df_clean = clean_data(df, target_col='target')
    print("\nOutput:\n", df_clean)
>>>>>>> f80310cf3a2ef8383f0d05dcca483e9bcc64aa12
