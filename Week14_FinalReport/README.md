# Week 14 Final Project – Feature Importance Analysis

## Overview
This section of the project focuses on identifying key drivers of cybersecurity outcomes using supervised learning and explainable AI techniques.

## Methods Used
- Random Forest (Regression & Classification)
- k-Fold Cross-Validation (k=5)
- Feature Importance Analysis
- SHAP (Explainable AI)

## Targets Modeled
- Financial Loss (Regression)
- Resolution Time (Regression)
- Incident Severity (Classification)

## Key Outputs
Located in `outputs/`:
- Feature Importance Plots (3 models)
- SHAP Summary Plot
- Top 10 Feature CSV files
- Section 3 Report (text)

## Key Insight
Anomaly-based features (e.g., ocsvm_score, knn_distance_score) and incident scale (Number of Affected Users) consistently emerged as the most influential predictors, indicating that extreme and high-impact events drive cybersecurity outcomes.

## Notes
- Dataset is excluded from the repository due to size
- All outputs are reproducible by running:
  python feature_importance_analysis.py

## Author
Mark Nelson
