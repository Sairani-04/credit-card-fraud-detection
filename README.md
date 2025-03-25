# **Credit Card Fraud Detection (Anomaly Detection)**

## **Project Overview**

This project focuses on detecting fraudulent transactions in a highly imbalanced credit card dataset. Given that fraud cases are rare, anomaly detection techniques and ensemble learning methods are used to improve classification performance while minimizing false positives.

## **Problem Statement**

The goal is to develop a robust machine learning model that can accurately distinguish between fraudulent and legitimate transactions based on transaction features.

## **Dataset**

- The dataset contains **284,807 transactions**, with only **492 fraud cases (0.172%)**, making it highly imbalanced.
- Features are numerical and anonymized (V1, V2, ..., V28) due to confidentiality.
- The dataset includes **Time**, **Amount**, and a target label (**0: Not Fraud, 1: Fraud**).

## **Approach & Methodology**

### **1. Data Exploration & Preprocessing**

- Handled class imbalance using **undersampling/oversampling techniques**.
- Scaled features using **StandardScaler** to improve model convergence.
- Explored feature distributions to identify patterns in fraudulent transactions.

### **2. Model Training & Evaluation**

- Implemented multiple supervised models:
  - **Logistic Regression**
  - **Random Forest Classifier**
  - **XGBoost Classifier**
  - **Stacking Classifier (Random Forest + XGBoost with Logistic Regression as meta-classifier)**
- Evaluated models using:
  - **Precision, Recall, F1-score** (to assess fraud detection capability)
  - **ROC-AUC Score** (to measure model discrimination power)

### **3. Ensemble Learning**

- Combined **Random Forest** and **XGBoost** in a stacking classifier for better fraud detection.
- Optimized decision threshold to **minimize false negatives while maintaining a low false positive rate**.

## **Results & Key Findings**

- **Stacking Classifier (RF + XGBoost + Logistic Regression)** achieved the best results:
  - **ROC-AUC Score: \~0.93**
  - **F1-score for Fraud: \~0.85** (improved recall & precision)
- Lowered false negatives, ensuring fraudulent transactions are caught effectively.

## **Future Improvements**

- Experiment with **autoencoders and deep learning** for anomaly detection.
- Implement **real-time fraud detection system**.
- Explore **cost-sensitive learning** to better balance fraud detection with minimal business disruption.

## **How to Run the Notebook**

1. Install required libraries:
   ```bash
   pip install numpy pandas scikit-learn xgboost matplotlib seaborn
   ```
2. Run `fraud_detection.ipynb` in Jupyter Notebook.
3. Train the model and evaluate results.

## **Conclusion**

This project demonstrates how **ensemble learning and anomaly detection** techniques can significantly improve fraud detection in highly imbalanced datasets. The stacking classifier approach proved effective in **reducing false negatives** while maintaining high overall accuracy.

