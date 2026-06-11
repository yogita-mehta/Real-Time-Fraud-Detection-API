"""
model.py  –  Train a Random Forest fraud-detection model on synthetic data.
Run this script FIRST before starting the API server.
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib

# ── 1. Generate synthetic, highly imbalanced credit-card dataset ─────────────
print("[1/4] Generating synthetic dataset (50 000 rows, ~1 % fraud)...")

X, y = make_classification(
    n_samples=50_000,
    n_features=20,
    n_informative=15,
    n_redundant=2,
    n_clusters_per_class=2,
    weights=[0.99, 0.01],   # 99 % legitimate, 1 % fraud
    flip_y=0.001,
    random_state=42,
)

feature_names = (
    ["amount", "hour_of_day", "day_of_week"]
    + [f"v{i}" for i in range(1, 18)]   # 17 anonymised PCA-like features
)
df = pd.DataFrame(X, columns=feature_names)
df["is_fraud"] = y

print(f"    Total rows  : {len(df):,}")
print(f"    Fraud rows  : {df['is_fraud'].sum():,}  ({df['is_fraud'].mean()*100:.2f} %)")

# ── 2. Train / test split ────────────────────────────────────────────────────
print("[2/4] Splitting data (80 / 20)...")
X_train, X_test, y_train, y_test = train_test_split(
    df[feature_names], df["is_fraud"],
    test_size=0.20,
    stratify=df["is_fraud"],
    random_state=42,
)

# ── 3. Train model ───────────────────────────────────────────────────────────
print("[3/4] Training Random Forest (class_weight=\'balanced\') ...")
clf = RandomForestClassifier(
    n_estimators=150,
    max_depth=12,
    class_weight="balanced",   # handles class imbalance automatically
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)

# ── 4. Evaluate ──────────────────────────────────────────────────────────────
print("[4/4] Evaluating model on held-out test set...")
y_prob = clf.predict_proba(X_test)[:, 1]
y_pred = clf.predict(X_test)

roc_auc = roc_auc_score(y_test, y_prob)
print(f"\n  ROC-AUC Score : {roc_auc:.4f}")
print("\n  Classification Report:")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

# ── 5. Persist artefact ──────────────────────────────────────────────────────
ARTEFACT = {
    "model": clf,
    "feature_names": feature_names,
}
joblib.dump(ARTEFACT, "fraud_model.pkl")
print("  Model saved → fraud_model.pkl")
