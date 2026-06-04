# Oxford Battery Degradation Analysis & SOH Predictor

This repository contains an end-to-end machine learning pipeline to analyze and predict battery State of Health (SOH) using the Oxford Battery Degradation Dataset. It transitions from exploratory data analysis and model optimization to model explainability and a deployment-ready GUI.

<img width="1920" height="1020" alt="Screenshot 2026-06-03 135811" src="https://github.com/user-attachments/assets/157057f6-f371-4c50-9ac3-f7fca2944b09" />

## Features

- **Exploratory Data Analysis (EDA):** Visualizes feature correlations, feature-to-target relationships, and battery degradation profiles.
- **Optimized Machine Learning Pipeline:** - Standardizes features using a robust preprocessing pipeline.
  - Explores multiple estimators (`Ridge`, `RandomForestRegressor`, `GradientBoostingRegressor`) using `RandomizedSearchCV`.
  - Performs fine-tuning on the top-performing model via `GridSearchCV`.
- **Model Evaluation:** Computes regression metrics ($MAE$, $MSE$, $R^2$) and translates performance into binary classification evaluation (ROC-AUC) using a median threshold.
- **Model Explainability:** Implements global and local feature importance interpretations via SHAP values.
- **Interactive GUI application:** Built-in `Tkinter` desktop application interface allowing deployment-ready inference and visual validation tracking.

## Getting Started

### Prerequisites

Ensure you have Python 3.10+ installed along with the following libraries:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn shap
