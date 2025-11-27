import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

# Load data
train_df = pd.read_csv("train (1).csv")
test_df = pd.read_csv("test (1).csv")

y = train_df["has_copd_risk"].astype(int)
X = train_df.drop(columns=["has_copd_risk", "patient_id"])
X_test = test_df.drop(columns=["patient_id"])

# BASIC PREPROCESSING
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Impute
X[num_cols] = SimpleImputer(strategy="median").fit_transform(X[num_cols])
X[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])

# Label encode categoricals
le = LabelEncoder()
for c in cat_cols:
    X[c] = le.fit_transform(X[c].astype(str))
    X_test[c] = le.transform(X_test[c].astype(str))

# -------------------------------
#   RANDOM FOREST CROSS-VALIDATION
# -------------------------------
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf = RandomForestClassifier(
    n_estimators=600,
    max_depth=18,
    min_samples_split=4,
    min_samples_leaf=2,
    max_features="sqrt",
    bootstrap=True,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

oof_preds = np.zeros(len(X))
for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    rf.fit(X_tr, y_tr)
    proba = rf.predict_proba(X_val)[:, 1]
    oof_preds[val_idx] = proba

# FIND OPTIMAL THRESHOLD
best_f1 = 0
best_thr = 0.5

for t in np.linspace(0.2, 0.8, 61):
    preds = (oof_preds > t).astype(int)
    score = f1_score(y, preds)
    if score > best_f1:
        best_f1 = score
        best_thr = t

print("Best OOF F1:", best_f1)
print("Best threshold:", best_thr)

# FINAL MODEL TRAIN
rf.fit(X, y)
test_probs = rf.predict_proba(X_test)[:, 1]
test_preds = (test_probs > best_thr).astype(int)

# CREATE SUBMISSION
submission = pd.DataFrame({
    "patient_id": test_df["patient_id"],
    "has_copd_risk": test_preds
})

submission.to_csv("rf_optimized_submission.csv", index=False)
print("Saved rf_optimized_submission.csv")
