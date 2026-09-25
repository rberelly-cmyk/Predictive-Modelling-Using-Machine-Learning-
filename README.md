# Credit Card Default Prediction Using Machine Learning

## Project Overview

This project focuses on predicting whether a credit card customer is likely to default on their next payment using machine learning.

The project analyses historical credit card customer data to identify patterns in repayment behaviour, credit limits, billing amounts, and payment history.

The main goal is to build and compare different machine learning classification models and understand how they can be used for credit risk assessment.

## Problem Statement

Credit card default is a major concern for financial institutions because missed payments can result in financial losses.

This project uses machine learning algorithms to predict the likelihood of a customer defaulting on their next payment based on historical financial and demographic information.

The task is treated as a binary classification problem:

- `0` – No default on the next payment.
- `1` – Default on the next payment.

## Objectives

- Perform exploratory data analysis (EDA).
- Clean and preprocess the dataset.
- Analyse customer demographics and repayment patterns.
- Train multiple machine learning classification models.
- Evaluate models using different performance metrics.
- Compare model performance and apply hyperparameter tuning.
- Identify important features influencing predictions.
- Discuss limitations and possible future improvements.

## Dataset

The project uses the `Credit_Card.csv` dataset, which contains historical credit card customer information.

Important features include:

| Feature | Description |
|---|---|
| LIMIT_BAL | Credit limit granted to the customer |
| AGE | Customer age |
| PAY_0, PAY_2–PAY_6 | Historical repayment status |
| BILL_AMT1–BILL_AMT6 | Previous billing statement amounts |
| PAY_AMT1–PAY_AMT6 | Previous payment amounts |
| SEX | Customer gender code |
| EDUCATION | Education category |
| MARRIAGE | Marital status category |

**Target variable:** `default.payment.next.month`

The target variable indicates whether a customer defaulted on the next payment.

## Technologies Used

- Python
- Jupyter Notebook
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- SHAP (optional, for model interpretation)

## Machine Learning Models

The following classification models are implemented and compared:

1. Logistic Regression
2. Support Vector Classifier (SVC)
3. Decision Tree Classifier
4. Random Forest Classifier
5. K-Nearest Neighbours (KNN)
6. HistGradientBoostingClassifier

A Dummy Classifier is also used as a baseline to compare the performance of the trained models against a simple prediction strategy.

## Project Workflow

The project follows these main steps:

1. Load and inspect the dataset.
2. Perform exploratory data analysis and visualisation.
3. Clean and preprocess the data.
4. Prepare input features and the target variable.
5. Split the data into training and testing sets.
6. Train multiple classification models.
7. Evaluate and compare model performance.
8. Perform hyperparameter tuning on selected models.
9. Analyse feature importance and model interpretability.
10. Summarise findings, limitations, and future improvements.

## Model Evaluation

The models are evaluated using the following metrics:

| Metric | Purpose |
|---|---|
| Accuracy | Measures the proportion of correctly classified customers |
| Precision | Measures the proportion of predicted defaulters who actually defaulted |
| Recall | Measures the proportion of actual defaulters correctly identified |
| F1-score | Balances precision and recall |
| ROC-AUC | Measures the ability to distinguish between the two classes |
| Average Precision | Summarises precision-recall performance |

Since credit card default datasets can contain imbalanced classes, recall, F1-score, ROC-AUC, and Average Precision are considered alongside accuracy.

The models are compared using their actual test-set results.

## Model Interpretability

Feature importance and permutation importance are used to understand which input variables contribute to model predictions.

SHAP-based interpretation is also available as an optional approach, depending on package installation and model compatibility.

These methods help explain model behaviour but do not establish causal relationships.

## Repository Structure

```text
Credit-Card-Default-Prediction/
│
├── Credit Card Default Prediction.ipynb
├── Credit_Card.csv
└── README.md
