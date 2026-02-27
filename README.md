# 📌 CTR Prediction System

## 📖 Overview

This project implements a **Click-Through Rate (CTR) Prediction Pipeline** using machine learning techniques.

The objective is to predict the probability of a user clicking on an advertisement based on structured feature inputs.

CTR prediction is widely used in:

- Digital Advertising
- Recommendation Systems
- Performance Marketing
- Real-time Bidding Systems

---

## 🧠 Problem Statement

Given user and advertisement features, predict:

> **P(click = 1 | features)**

Since CTR prediction is a probability estimation task, the focus is on:

- **AUC-ROC**
- **Log Loss**
- Proper probability calibration

Accuracy is not prioritized due to class imbalance in click data.

---

---

## ⚙️ Technologies Used

- Python
- Pandas & NumPy
- Scikit-learn
- Joblib (Model Serialization)

---

## 🛠️ Pipeline Workflow

### 1️⃣ Data Preprocessing

- Categorical feature encoding using OneHotEncoder
- Numerical feature passthrough
- Stratified train-test split

### 2️⃣ Model Training

Two models were trained:

- Logistic Regression (Baseline)
- Random Forest Classifier

### 3️⃣ Evaluation Metrics

- **AUC-ROC**
- **Log Loss**
- Accuracy (for reference only)

---

## 📊 Model Results

### Logistic Regression

- AUC: 0.67
- Log Loss: 0.239
- Accuracy: 0.93

### Random Forest

- AUC: 0.61
- Log Loss: 0.58
- Accuracy: 0.92

### ✅ Conclusion

Logistic Regression outperformed Random Forest in terms of AUC and Log Loss, indicating better probability calibration and ranking performance.

For CTR prediction tasks, linear models often perform competitively due to high-dimensional sparse features.

---

## 🚀 How to Run

### 1️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```
