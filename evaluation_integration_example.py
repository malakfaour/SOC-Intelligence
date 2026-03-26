#!/usr/bin/env python
"""
TabNet Evaluation Integration Example

Shows how to use evaluation metrics with actual TabNet model predictions
for both triage and remediation tasks.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path.cwd() / "src"))

print("\n" + "=" * 70)
print("TABNET EVALUATION INTEGRATION EXAMPLE")
print("=" * 70)

# ============================================================================
# Example 1: Triage Evaluation with TabNet
# ============================================================================
print("\n[EXAMPLE 1] Triage Evaluation")
print("-" * 70)

print("""
Scenario: Evaluate a TabNet triage model on test set

Steps:
  1. Load trained TabNet model and test data
  2. Generate triage predictions
  3. Evaluate with macro-F1, per-class metrics, confusion matrix
  4. Save results to reports/metrics/triage_metrics.json
""")

print("""
Code Example:

from src.evaluation.metrics import evaluate_tabnet_triage
from src.models.tabnet.train import load_tabnet_triage_model
from src.training.train_tabnet import load_tabnet_data

# Load data
X_train, X_val, X_test, y_train, y_val, y_test = load_tabnet_data()

# Load trained model
model, results = train_tabnet_triage_model(
    X_train, X_val, X_test, 
    y_train, y_val, y_test
)

# Get predictions  
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# Evaluate
metrics, formatted = evaluate_tabnet_triage(
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba,
    output_dir="reports/metrics"
)

# Display results
print(formatted)

# Access individual metrics
print(f"Macro-F1: {metrics['macro_f1']:.4f}")
print(f"Accuracy: {metrics['overall_accuracy']:.4f}")
for class_name, class_metrics in metrics["per_class_metrics"].items():
    print(f"  {class_name}: P={class_metrics['precision']:.3f}, "
          f"R={class_metrics['recall']:.3f}, F1={class_metrics['f1']:.3f}")
""")

# ============================================================================
# Example 2: Remediation Evaluation with TabNet
# ============================================================================
print("\n[EXAMPLE 2] Remediation Evaluation")
print("-" * 70)

print("""
Scenario: Evaluate TabNet multi-label remediation predictions

Steps:
  1. Generate multi-label predictions (binary for each remediation)
  2. Evaluate with Hamming loss and per-label F1
  3. Identify best/worst performing remediation actions
  4. Save results to reports/metrics/remediation_metrics.json
""")

print("""
Code Example:

from src.evaluation.metrics import evaluate_tabnet_remediation
from src.models.tabnet.multitask import MultiTaskTabNet

# Load data
X_test = ...  # test features
y_test_remediation = ...  # true multi-label targets (n_samples, n_remediations)

# Load trained multi-task model
model = MultiTaskTabNet(n_features=44, n_triage_classes=3, n_remediations=10)
# ... load model weights ...

# Get predictions
remediation_logits = model.predict_remediations(X_test)
y_pred_remediation = (remediation_logits > 0.5).astype(int)  # Threshold at 0.5

# Evaluate
metrics, formatted = evaluate_tabnet_remediation(
    y_true=y_test_remediation,
    y_pred=y_pred_remediation,
    output_dir="reports/metrics"
)

# Display results
print(formatted)

# Access individual metrics
print(f"Hamming Loss: {metrics['hamming_loss']:.4f}")
print(f"Macro-F1: {metrics['macro_f1']:.4f}")
print(f"Micro-F1: {metrics['micro_f1']:.4f}")

# Find best performing remediations
best_remediation = max(metrics['per_label_f1'].items(), key=lambda x: x[1])
print(f"Best remediation: {best_remediation[0]} (F1={best_remediation[1]:.4f})")
""")

# ============================================================================
# Example 3: Combined Triage + Remediation Evaluation
# ============================================================================
print("\n[EXAMPLE 3] Combined Multi-Task Evaluation")
print("-" * 70)

print("""
Scenario: Evaluate complete multi-task TabNet model

Steps:
  1. Get triage AND remediation predictions
  2. Evaluate both tasks in sequence
  3. Generate comprehensive report
  4. Compare task performance
""")

print("""
Code Example:

from src.evaluation.metrics import (
    evaluate_tabnet_triage, 
    evaluate_tabnet_remediation
)
from src.models.tabnet.multitask import MultiTaskTabNet

# Load multi-task model  
model = MultiTaskTabNet(...)

# Get predictions
triage_logits, remediation_logits = model(X_test)

# Convert to class predictions
y_triage_pred = np.argmax(triage_logits, axis=1)
y_remediation_pred = (remediation_logits > 0.5).astype(int)

# Evaluate both tasks
triage_metrics, triage_formatted = evaluate_tabnet_triage(
    y_test_triage, y_triage_pred, 
    output_dir="reports/metrics"
)

remediation_metrics, remediation_formatted = evaluate_tabnet_remediation(
    y_test_remediation, y_remediation_pred,
    output_dir="reports/metrics"
)

# Display full report
print(triage_formatted)
print(remediation_formatted)

# Generate summary
print("MULTI-TASK EVALUATION SUMMARY")
print(f"Triage Performance:      Macro-F1 = {triage_metrics['macro_f1']}")
print(f"Remediation Performance: Hamming Loss = {remediation_metrics['hamming_loss']}")
print()
print("Files saved to reports/metrics/:")
print("  - triage_metrics.json")
print("  - remediation_metrics.json")
""")

# ============================================================================
# Example 4: Programmatic Access to Evaluators
# ============================================================================
print("\n[EXAMPLE 4] Using Evaluator Classes Directly")
print("-" * 70)

print("""
Advanced Usage: Direct access to evaluator classes for custom workflows

Code Example:

from src.evaluation.metrics import TriageEvaluator, RemediationEvaluator

# Initialize evaluators
triage_eval = TriageEvaluator(n_classes=3)
remediation_eval = RemediationEvaluator(n_remediations=10)

# Compute metrics
triage_metrics = triage_eval.compute_metrics(y_true, y_pred)
remediation_metrics = remediation_eval.compute_metrics(y_true, y_pred)

# Format for display
triage_formatted = triage_eval.format_results(triage_metrics)
remediation_formatted = remediation_eval.format_results(remediation_metrics)

# Save results manually
from src.evaluation.metrics import save_triage_metrics, save_remediation_metrics

save_triage_metrics(triage_metrics, output_dir="reports/metrics")
save_remediation_metrics(remediation_metrics, output_dir="reports/metrics")

# Access raw metrics for further analysis
macro_f1 = triage_metrics["macro_f1"]
hamming_loss = remediation_metrics["hamming_loss"]

# Iterate over per-class results
for class_name, metrics_dict in triage_metrics["per_class_metrics"].items():
    print(f"{class_name}: F1={metrics_dict['f1']:.4f}")
""")

# ============================================================================
# Output Structure
# ============================================================================
print("\n[DATA FORMAT] Saved Metrics Files")
print("-" * 70)

print("""
Triage Metrics (reports/metrics/triage_metrics.json):
  {
    "macro_f1": float,
    "overall_accuracy": float,
    "per_class_metrics": {
      "Class_0": {"precision": float, "recall": float, "f1": float, "support": int},
      "Class_1": {...},
      "Class_2": {...}
    },
    "confusion_matrix": [[...], [...], [...]]
  }

Remediation Metrics (reports/metrics/remediation_metrics.json):
  {
    "hamming_loss": float,
    "micro_f1": float,
    "macro_f1": float,
    "per_label_f1": {
      "Remediation_0": float,
      "Remediation_1": float,
      ...
      "Remediation_9": float
    },
    "label_support": {
      "Remediation_0": int,
      "Remediation_1": int,
      ...
    }
  }
""")

# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 70)
print("✓ EVALUATION METRICS READY FOR USE")
print("=" * 70)

print("""
Available Functions:
  ✓ evaluate_tabnet_triage()      - High-level triage evaluation
  ✓ evaluate_tabnet_remediation() - High-level remediation evaluation
  ✓ TriageEvaluator               - Low-level triage evaluator class
  ✓ RemediationEvaluator          - Low-level remediation evaluator class
  ✓ save_triage_metrics()         - Save triage metrics to JSON
  ✓ save_remediation_metrics()    - Save remediation metrics to JSON

Metrics Computed:
  
  Triage:
    ✓ Macro-F1 score
    ✓ Per-class precision, recall, F1
    ✓ Confusion matrix
    ✓ Overall accuracy

  Remediation:
    ✓ Hamming loss (fraction of incorrect labels)
    ✓ Per-label F1 scores
    ✓ Micro and macro-averaged F1
    ✓ Label support (positive samples per label)

Output Location:
  reports/metrics/
    - triage_metrics.json
    - remediation_metrics.json
""")

print("✓ Ready to evaluate TabNet models!")
