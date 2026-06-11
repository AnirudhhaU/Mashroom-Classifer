# Mushroom Classification — Edible vs Poisonous

A Kaggle competition ML project that classifies mushrooms as edible or poisonous using an ensemble of tree-based models, achieving **99.28% accuracy**.

## Overview

The dataset contains large-scale mushroom records with categorical features like odor, gill color, spore print color, and stalk root. The key challenge is handling missing values correctly — naively imputing them destroys signal carried by missingness itself.

## Approach

### 1. Missing Value Handling
- Categorical `NaN` → filled with `'unknown'` (treated as its own category, not imputed)
- `'missing'` string in `stalk-root` kept as-is — it carries real predictive signal
- Numerical `NaN` → median from training set

### 2. Feature Engineering
- Missingness indicator flags (`odor_is_unknown`, `stalkroot_is_missing`)
- Interaction features between strongest predictors (`odor × gill-color`, `odor × spore-print-color`, etc.)
- Combined train+test label encoding to ensure consistent category mapping

### 3. Models Benchmarked (10 classifiers, 5-fold CV)
Logistic Regression, Decision Tree, Random Forest, SVM, KNN, AdaBoost, Gradient Boosting, XGBoost, LightGBM, Extra Trees

### 4. Hyperparameter Tuning
Top 3 models (Random Forest, XGBoost, LightGBM) tuned with `RandomizedSearchCV` + `StratifiedKFold`

### 5. Ensemble
Soft-voting ensemble of tuned RF + XGBoost + LightGBM, retrained on full data for final submission

## Results

| Model | Val Accuracy |
|---|---|
| Soft Voting Ensemble (RF + XGB + LGBM) | **99.28%** |
| LightGBM (tuned) | ~99.1% |
| XGBoost (tuned) | ~99.0% |
| Random Forest (tuned) | ~98.8% |

## Files

```
mushroom_classification_v3_final.ipynb   # Full pipeline notebook
```

## Setup

This project runs on Kaggle. To run locally:

```bash
pip install numpy pandas scikit-learn xgboost lightgbm matplotlib seaborn
jupyter notebook mushroom_classification_v3_final.ipynb
```

> **Note:** Update the data paths in the notebook — replace `/kaggle/input/...` with your local CSV paths.

## Tech Stack

Python · scikit-learn · XGBoost · LightGBM · pandas · NumPy · Matplotlib · Seaborn
