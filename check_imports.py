#!/usr/bin/env python
"""Check import paths quickly"""

import sys

print("\n[Checking imports...]")

try:
    print("  Checking pytorch_tabnet...")
    import pytorch_tabnet
    print(f"    ✓ pytorch_tabnet found at: {pytorch_tabnet.__file__}")
    print(f"    Available: {[x for x in dir(pytorch_tabnet) if not x.startswith('_')]}")
except ImportError as e:
    print(f"    ✗ pytorch_tabnet: {e}")

try:
    print("  Checking torch...")
    import torch
    print(f"    ✓ torch {torch.__version__}")
except ImportError as e:
    print(f"    ✗ torch: {e}")

try:
    print("  Checking optuna...")
    import optuna
    print(f"    ✓ optuna {optuna.__version__}")
except ImportError as e:
    print(f"    ✗ optuna: {e}")

try:
    print("  Checking sklearn...")
    import sklearn
    print(f"    ✓ sklearn {sklearn.__version__}")
except ImportError as e:
    print(f"    ✗ sklearn: {e}")

print("\n[Attempting TabNetClassifier import...]")
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    print("  ✓ TabNetClassifier imported from pytorch_tabnet.tab_model")
except Exception as e:
    print(f"  ✗ Failed: {e}")

print("\nDone!")
