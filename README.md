<div align="center">

# 📉 Customer Churn Prediction & Explainability Engine

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg?logo=python&logoColor=white)](https://python.org)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-111111?logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-ff69b4.svg)](https://shap.readthedocs.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An end-to-end Machine Learning pipeline that predicts customer churn risk and provides interpretable, feature-level insights using SHAP values.*

[Key Features](#-key-features) • [SHAP Integration](#-explainability-with-shap) • [Installation](#-quick-start) • [Model Benchmarks](#-model-performance)

</div>

---

## 📌 Project Overview

Customer retention is critical for subscription and service-based business models. This repository implements an end-to-end machine learning solution designed to:
* Classify customers into **Low Risk** vs. **High Risk** churn categories.
* Handle severe class imbalance using advanced resampling strategies (**SMOTE**).
* Provide **Explainable AI (XAI)** attributions for every single prediction so retention teams know *why* a customer is leaving.

---

## ✨ Key Features

- 🧹 **Automated Preprocessing Pipeline:** Missing value imputation, categorical encoding, and feature scaling.
- ⚖️ **Imbalance Handling:** Synthetic Minority Over-sampling Technique (SMOTE) integrated into cross-validation.
- ⚡ **Multi-Model Evaluation:** Benchmarked Logistic Regression, Random Forest, and XGBoost Classifiers.
- 🔍 **Model Interpretability:** Native integration with **SHAP** for global summary plots and local waterfall diagnostics.

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | 
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.79 | 0.62 | 0.54 |
| Random Forest | 0.81 | 0.65 | 0.58|
| **Gradient Boosting(XGBoost) (Best)** | **0.92** | **0.89** | **0.86** |

---

## 💡 Explainability with SHAP

Rather than treating the churn model as a black box, this repository uses **SHAP (SHapley Additive exPlanations)** to break down predictions into exact numerical contributions:

$$\text{Prediction} = \text{Base Value} + \sum \text{SHAP Value}_i$$

<details>
<summary>🔍 <b>Click to view SHAP Python Implementation Snippet</b></summary>

```python
import shap
import xgboost as xgb

# Load trained XGBoost model and dataset
model = xgb.XGBClassifier()
model.load_model("models/best_xgboost_churn.json")

# Initialize SHAP TreeExplainer
explainer = shap.TreeExplainer(model)
shap_values = explainer(X_test)

# 1. Global Summary Plot (Overview of all features)
shap.summary_plot(shap_values, X_test)

# 2. Local Waterfall Plot (Single customer risk breakdown)
shap.plots.waterfall(shap_values[0])
