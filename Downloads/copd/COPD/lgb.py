import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
import lightgbm as lgb

# ------------------
# Load (read CSVs)
# ------------------
train = pd.read_csv(r"c:\Users\LEGION\OneDrive\Desktop\ML2.1\train (1).csv")
test  = pd.read_csv(r"c:\Users\LEGION\OneDrive\Desktop\ML2.1\test (1).csv")

y = train["has_copd_risk"].astype(int)
train_id = train["patient_id"]
test_id  = test["patient_id"]

X = train.drop(columns=["patient_id", "has_copd_risk"])
X_test = test.drop(columns=["patient_id"])

# ------------------
# Convert categoricals properly
# ------------------
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

for c in cat_cols:
    X[c] = X[c].astype("category")
    X_test[c] = X_test[c].astype("category")

# ------------------
# Model config
# ------------------
params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 64,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "bagging_freq": 1,
    "verbosity": -1,
}

# ------------------
# CV
# ------------------
NFOLD = 5
skf = StratifiedKFold(n_splits=NFOLD, shuffle=True, random_state=42)

oof = np.zeros(len(X))
test_preds = np.zeros((len(X_test), NFOLD))

for i, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
    print(f"Fold {i+1}/{NFOLD}")

    X_train, X_val = X.iloc[train_idx], X.iloc[valid_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[valid_idx]

    train_set = lgb.Dataset(X_train, y_train, categorical_feature=cat_cols)
    val_set   = lgb.Dataset(X_val, y_val, categorical_feature=cat_cols)

    model = lgb.train(
        params,
        train_set,
        num_boost_round=2000,
        valid_sets=[val_set],
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )

    oof[valid_idx] = model.predict(X_val, num_iteration=model.best_iteration)
    test_preds[:, i] = model.predict(X_test, num_iteration=model.best_iteration)

# ------------------
# Threshold tuning
# ------------------
best_f1 = -1
best_t = 0.5

for t in np.linspace(0.1, 0.9, 81):
    f = f1_score(y, (oof > t).astype(int))
    if f > best_f1:
        best_f1 = f
        best_t = t

print("Best threshold:", best_t, "F1:", best_f1)

# ------------------
# Final test preds
# ------------------
final_preds = (test_preds.mean(axis=1) > best_t).astype(int)

submission = pd.DataFrame({
    "patient_id": test_id,
    "has_copd_risk": final_preds
})

submission.to_csv("submission_lgb_native_cat-1.csv", index=False)
print("Saved submission_lgb_native_cat.csv")
