# Credit Card Fraud Detection (Anomaly Detection)

## Problem Statement

Detect fraudulent transactions in credit card data using machine learning techniques while handling class imbalance.

## Dataset

The dataset is sourced from **[Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)**.

### How to Get the Dataset

Since the dataset is too large to upload to GitHub, follow these steps:

1. Download the dataset manually from the [Kaggle link](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud).
2. Extract the `creditcard.csv` file and place it in the project directory.

## Project Overview

This project builds an ensemble model combining **Random Forest** and **XGBoost** to improve fraud detection. It follows these steps:

- **Importing Libraries**: Load essential Python libraries for data processing, visualization, and modeling.
- **Load Data**: Read and inspect the dataset.
- **Exploratory Data Analysis (EDA)**:
  - Visualize feature distributions.
  - Identify class imbalance in fraud vs. non-fraud transactions.
- **Data Preprocessing**:
  - Scale numerical features.
  - Handle imbalanced data using **undersampling/oversampling**.
- **Train Supervised Learning Models**:
  - **Random Forest**
  - **XGBoost**
- **Ensemble Model (XGBoost + Random Forest)**:
  - Combine models using **StackingClassifier** with a **Logistic Regression** meta-learner.
- **Evaluation**:
  - Classification report (Precision, Recall, F1-score).
  - **ROC-AUC Score** to measure performance.

## Results

The ensemble model achieves an **ROC-AUC score of 0.93**, significantly improving fraud detection while maintaining high precision and recall.

## How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Jupyter Notebook:
   ```bash
   jupyter notebook
   ```
3. Open `fraud_detection.ipynb` and execute the cells step by step.

## Requirements

Ensure you have the following libraries installed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

## License

This project is for educational purposes only. Dataset credits go to the original authors on Kaggle.


