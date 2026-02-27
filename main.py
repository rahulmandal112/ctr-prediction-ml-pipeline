import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, log_loss, accuracy_score
import joblib

# ==============================
# 1. Load Dataset
# ==============================

# Replace with your dataset path
df = pd.read_csv("ctr_dataset.csv")

# Assume target column is 'clicked' (0/1)
target = "clicked"

X = df.drop(columns=[target])
y = df[target]

# ==============================
# 2. Identify Feature Types
# ==============================

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

# ==============================
# 3. Preprocessing Pipeline
# ==============================

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

# ==============================
# 4. Train/Test Split
# ==============================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==============================
# 5. Models
# ==============================

log_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]
)

rf_model = Pipeline(
    steps=[
        ("preprocessing", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

# ==============================
# 6. Train Logistic Regression
# ==============================

log_model.fit(X_train, y_train)
log_preds = log_model.predict_proba(X_test)[:, 1]

print("Logistic Regression Results")
print("AUC:", roc_auc_score(y_test, log_preds))
print("Log Loss:", log_loss(y_test, log_preds))
print("Accuracy:", accuracy_score(y_test, (log_preds > 0.5)))

print("\n--------------------------------\n")

# ==============================
# 7. Train Random Forest
# ==============================

rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict_proba(X_test)[:, 1]

print("Random Forest Results")
print("AUC:", roc_auc_score(y_test, rf_preds))
print("Log Loss:", log_loss(y_test, rf_preds))
print("Accuracy:", accuracy_score(y_test, (rf_preds > 0.5)))

# ==============================
# 8. Save Best Model
# ==============================

joblib.dump(rf_model, "ctr_model.pkl")
print("\nModel saved as ctr_model.pkl")