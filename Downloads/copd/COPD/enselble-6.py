# stacking_ensemble.py
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier, early_stopping
from xgboost import XGBClassifier
from sklearn.metrics import f1_score, precision_recall_curve

# --- 1) Load data ---
train_df = pd.read_csv('c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\train (1).csv')
test_df = pd.read_csv('c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\test (1).csv')

y = train_df['has_copd_risk'].astype(int)
X = train_df.drop(columns=['has_copd_risk', 'patient_id'])
X_test = test_df.drop(columns=['patient_id'])

# --- 2) Preprocessing setup (same transform for all models) ---
num_cols = X.select_dtypes(include=['int64','float64']).columns.tolist()
cat_cols = X.select_dtypes(include=['object']).columns.tolist()

num_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transform = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', num_transform, num_cols),
    ('cat', cat_transform, cat_cols)
], sparse_threshold=0)  # return dense output for stacking

# --- 3) Base models (use sensible regularization/params to avoid overfit) ---
base_models = [
    ('lgb', LGBMClassifier(n_estimators=1000, learning_rate=0.05, 
                           num_leaves=31, colsample_bytree=0.8,
                           subsample=0.8, random_state=42)),
    ('xgb', XGBClassifier(n_estimators=1000, learning_rate=0.05, 
                          max_depth=6, subsample=0.8, colsample_bytree=0.8,
                          use_label_encoder=False, eval_metric='logloss',
                          random_state=42)),
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=12, 
                                  min_samples_leaf=3, random_state=42, n_jobs=-1))
]

# --- 4) OOF stacking function ---
def get_oof_preds(models, X, y, X_test, n_splits=5, random_state=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    n_models = len(models)
    oof_train = np.zeros((X.shape[0], n_models))
    oof_test = np.zeros((X_test.shape[0], n_models))
    for i, (name, clf) in enumerate(models):
        print(f"OOF for {name}")
        oof_test_single = np.zeros((X_test.shape[0], n_splits))
        for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y)):
            X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
            y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]
            pipe = Pipeline([
                ('pre', preprocessor),
                ('clf', clf)
            ])
            # fit with early stopping only for LGB
            if name == 'lgb':
                # Preprocess validation data separately
                X_val_processed = pipe.named_steps['pre'].fit_transform(X_tr)
                X_val_processed = pipe.named_steps['pre'].transform(X_val)
                pipe.fit(X_tr, y_tr,
                         clf__eval_set=[(X_val_processed, y_val)],
                         clf__callbacks=[early_stopping(50)])
            else:
                pipe.fit(X_tr, y_tr)
            # proba for positive class
            oof_train[val_idx, i] = pipe.predict_proba(X_val)[:,1]
            oof_test_single[:, fold] = pipe.predict_proba(X_test)[:,1]
            print(f"  fold {fold} done")
        oof_test[:, i] = oof_test_single.mean(axis=1)
    return oof_train, oof_test

# Run OOF stacking
oof_train, oof_test = get_oof_preds(base_models, X, y, X_test, n_splits=5)

# --- 5) Train meta-learner on OOF predictions ---
meta = LogisticRegression(max_iter=2000, class_weight='balanced', random_state=42)
meta.fit(oof_train, y)
oof_preds_meta_proba = meta.predict_proba(oof_train)[:,1]

# Find optimal threshold on OOF to maximize F1
def find_best_thresh(y_true, probs):
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 81):
        f = f1_score(y_true, (probs > t).astype(int))
        if f > best_f1:
            best_f1, best_t = f, t
    return best_t, best_f1

best_thr, best_f1 = find_best_thresh(y, oof_preds_meta_proba)
print("OOF Meta F1 (best threshold):", best_f1, "threshold:", best_thr)

# --- 6) Final test predictions ---
test_meta_proba = meta.predict_proba(oof_test)[:,1]
test_preds = (test_meta_proba > best_thr).astype(int)

# --- 7) Save submission ---
submission = pd.DataFrame({
    'patient_id': test_df['patient_id'],
    'has_copd_risk': test_preds
})
submission.to_csv('submission_stacked.csv', index=False)
print("Saved submission_stacked.csv")
