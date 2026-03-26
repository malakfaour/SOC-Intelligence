"""
TabNet Explainability Module - Quick Start Guide

This module provides comprehensive feature importance analysis and visualization
for TabNet models using attention masks from the model.explain() output.
"""

# ============================================================================
# EXAMPLE 1: Basic Usage - Generate All Explanations
# ============================================================================

from src.explainability.explainability import explain_tabnet_model, TabNetExplainer
import numpy as np

# Assuming you have a trained TabNet model and test data
# model = trained_tabnet_model
# X_test = test_features
# feature_names = list of feature names

# One-line analysis
results = explain_tabnet_model(
    model=model,
    X_test=X_test,
    feature_names=feature_names,
    output_dir="reports/figures",
    top_k=15,
    include_heatmap=True,
    include_step_plots=True
)

# ============================================================================
# Access Results
# ============================================================================

# Plot paths
print("Generated plots:")
for plot_name, plot_path in results['plots'].items():
    print(f"  - {plot_name}: {plot_path}")

# Top features
print("\nTop 5 Most Important Features:")
for i, (feature_name, importance) in enumerate(results['top_features'][:5], 1):
    print(f"  {i}. {feature_name}: {importance:.4f}")

# Model info
print(f"\nAnalysis Summary:")
print(f"  Samples: {results['n_samples']}")
print(f"  Features: {results['n_features']}")
print(f"  Steps: {results['n_steps']}")

# ============================================================================
# EXAMPLE 2: Advanced - Manual Control with TabNetExplainer
# ============================================================================

from src.explainability.explainability import (
    TabNetExplainer,
    plot_feature_importance,
    plot_step_importance,
    plot_mask_heatmap
)

# Initialize explainer
explainer = TabNetExplainer(model, feature_names=feature_names)

# Extract masks
feature_masks, predictions = explainer.get_feature_masks(X_test)
print(f"Mask shape: {feature_masks.shape}")  # (n_samples, n_steps, n_features)

# Compute importance
global_importance = explainer.aggregate_feature_importance(feature_masks, 'mean')
print(f"Global importance shape: {global_importance.shape}")  # (n_features,)

# Get step importance
step_importance = explainer.get_step_importance(feature_masks)
print(f"Step importance shape: {step_importance.shape}")  # (n_steps, n_features)

# Get top features
top_k_features = explainer.get_top_features(feature_masks, top_k=10)
print(f"Top 10 features: {top_k_features}")

# Generate individual plots
plot_feature_importance(
    explainer, feature_masks, top_k=15,
    output_path="reports/features_importance.png",
    title="Top 15 Most Important Features"
)

plot_step_importance(
    explainer, feature_masks, top_k=8,
    output_path="reports/step_importance.png",
    title="Feature Importance by Step"
)

plot_mask_heatmap(
    feature_masks,
    feature_names=feature_names,
    sample_indices=[0, 1, 2, 3, 4],  # First 5 samples
    output_path="reports/heatmap.png",
    title="Feature Masks Heatmap"
)

# ============================================================================
# EXAMPLE 3: Instance-Level Explanation
# ============================================================================

# Explain a specific instance
instance_explanation = explainer.explain_instance(X_test, instance_idx=0)

print("Instance 0 Explanation:")
print(f"  Prediction: {instance_explanation['prediction']}")
print(f"  Decision Steps: {len(instance_explanation['step_features'])}")

# Step-wise features
for step_name, step_features in instance_explanation['step_features'].items():
    print(f"\n  {step_name} - Top features:")
    for feature_name, importance in step_features[:3]:
        print(f"    - {feature_name}: {importance:.4f}")

# ============================================================================
# EXAMPLE 4: Custom Aggregation
# ============================================================================

# Different aggregation methods
mean_importance = explainer.aggregate_feature_importance(feature_masks, 'mean')
max_importance = explainer.aggregate_feature_importance(feature_masks, 'max')
sum_importance = explainer.aggregate_feature_importance(feature_masks, 'sum')

# Get top features with different methods
top_by_mean = explainer.get_top_features(feature_masks, top_k=10, aggregation='mean')
top_by_max = explainer.get_top_features(feature_masks, top_k=10, aggregation='max')

print("Top features comparison:")
print("  By Mean:", [f[0] for f in top_by_mean[:3]])
print("  By Max:", [f[0] for f in top_by_max[:3]])

# ============================================================================
# EXAMPLE 5: Batch Processing
# ============================================================================

# Process multiple test sets
test_sets = {'validation': X_val, 'test': X_test}
all_results = {}

for set_name, X_data in test_sets.items():
    results = explain_tabnet_model(
        model=model,
        X_test=X_data,
        feature_names=feature_names,
        output_dir=f"reports/{set_name}_analysis",
        top_k=15
    )
    all_results[set_name] = results

# Compare top features across sets
for set_name, results in all_results.items():
    print(f"\nTop 5 for {set_name}:")
    for feature_name, importance in results['top_features'][:5]:
        print(f"  {feature_name}: {importance:.4f}")

# ============================================================================
# OUTPUT STRUCTURE
# ============================================================================
"""
reports/figures/
├── feature_importance.png
│   - Horizontal bar chart of top-K most important features
│   - Color gradient from viridis colormap
│   - Value labels on each bar
│   - Sorted by importance
│
├── step_importance.png
│   - Subplots (one per decision step)
│   - Bar chart of top features per step
│   - Visualizes how feature importance changes through steps
│   - Helps understand model reasoning process
│
└── feature_mask_heatmap.png
    - 2D heatmap: samples x features
    - Color intensity = feature importance
    - Shows which features matter for which samples
    - First 10 samples by default
"""

# ============================================================================
# KEY CONCEPTS
# ============================================================================

"""
TabNet Attention Masks:
  - TabNet uses sequential attention masks during inference
  - Each step attends to specific features (sparse attention)
  - Masks shape: (n_samples, n_steps, n_features)
  - mask[i, step, j] = attention weight for feature j at step for sample i

Aggregation Methods:
  - 'mean': Average importance across all samples and steps
    Best for: Overall feature importance
  
  - 'max': Maximum importance across all samples and steps
    Best for: Features that matter for specific decisions
  
  - 'sum': Sum of importance across all samples and steps
    Best for: Total contribution magnitude

Features are ranked by their attention weights - higher weight = more important
"""

# ============================================================================
# TROUBLESHOOTING
# ============================================================================

"""
Q: "Model has no explain() method"
A: TabNetExplainer only works with pytorch_tabnet models
   (TabNetClassifier, TabNetRegressor)

Q: "Plot generation fails"
A: Make sure matplotlib and seaborn are installed:
   pip install matplotlib seaborn

Q: "Memory error with large datasets"
A: Process samples in batches:
   batch_size = 1000
   for i in range(0, len(X_test), batch_size):
       results = explain_tabnet_model(model, X_test[i:i+batch_size], ...)

Q: "Feature names not showing in plots"
A: Pass feature_names parameter to explain_tabnet_model():
   explainability.explain_tabnet_model(..., feature_names=['feat1', 'feat2'])

Q: "Heatmap too large/small"
A: Adjust sample_indices parameter in plot_mask_heatmap():
   plot_mask_heatmap(masks, sample_indices=list(range(5)))  # Show 5 samples
"""

# ============================================================================
# INTEGRATION PATTERNS
# ============================================================================

"""
1. End-of-Training Analysis
   After training, generate explanations for test set

2. Model Comparison
   Compare feature importance across models (XGBoost, LightGBM, TabNet)

3. Feature Engineering Validation
   Check if engineered features are actually used by model

4. Debugging Model Behavior
   Understand why model makes specific predictions

5. Business Stakeholder Reports
   Export visualizations for interpretation and documentation
"""

print(__doc__)
