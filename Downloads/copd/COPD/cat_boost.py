import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# load data
train = pd.read_csv("c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\train (1).csv")
test = pd.read_csv("c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\test (1).csv")

y = train["has_copd_risk"].astype(int)
X = train.drop(columns=["patient_id", "has_copd_risk"])
X_test = test.drop(columns=["patient_id"])

# identify categorical columns
cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

# cross-validation
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

oof_preds = np.zeros(len(X))

for train_idx, val_idx in skf.split(X, y):
    X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

    train_pool = Pool(X_tr, label=y_tr, cat_features=cat_cols)
    val_pool   = Pool(X_val, label=y_val, cat_features=cat_cols)

    model = CatBoostClassifier(
        iterations=2000,
        depth=7,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="F1",
        l2_leaf_reg=5,
        random_state=42,
        verbose=False
    )

    model.fit(train_pool, eval_set=val_pool, verbose=False)
    proba = model.predict_proba(val_pool)[:, 1]
    oof_preds[val_idx] = proba

# find best threshold
best_thr, best_f1 = 0.5, 0
for t in np.linspace(0.2, 0.8, 61):
    preds = (oof_preds > t).astype(int)
    f1 = f1_score(y, preds)
    if f1 > best_f1:
        best_f1 = f1
        best_thr = t

print("Best threshold:", best_thr)
print("OOF F1:", best_f1)

# final training
full_pool = Pool(X, label=y, cat_features=cat_cols)
test_pool = Pool(X_test, cat_features=cat_cols)

model.fit(full_pool, verbose=False)
test_preds = (model.predict_proba(test_pool)[:, 1] > best_thr).astype(int)

submission = pd.DataFrame({
    "patient_id": test["patient_id"],
    "has_copd_risk": test_preds
})
submission.to_csv("catboost_submission.csv", index=False)
print("Saved catboost_submission.csv")
