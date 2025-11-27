import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# ---------------------------
# Load Data
# ---------------------------
train_df = pd.read_csv('c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\train (1).csv')
test_df = pd.read_csv('c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\test (1).csv')

X = train_df.drop(columns=["has_copd_risk", "patient_id"])
y = train_df["has_copd_risk"].astype(int)
X_test = test_df.drop(columns=["patient_id"])

# ---------------------------
# Identify column types
# ---------------------------
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# ---------------------------
# Preprocessing
# ---------------------------
numeric_transform = SimpleImputer(strategy="median")
categorical_transform = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy="most_frequent")),
    ('onehot', OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transform, num_cols),
        ('cat', categorical_transform, cat_cols)
    ]
)

# ---------------------------
# Base Model (Random Forest)
# ---------------------------
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# ---------------------------
# FAST + EFFECTIVE hyperparameter search
# (Only the parameters that matter)
# ---------------------------
param_dist = {
    'clf__n_estimators': [400, 600, 800, 1000],
    'clf__max_depth': [10, 14, 18, None],
    'clf__min_samples_split': [2, 4, 6],
    'clf__min_samples_leaf': [1, 2, 3],
    'clf__max_features': ['sqrt', 0.4, 0.5, 0.7],
    'clf__class_weight': ['balanced'],  # IMPORTANT for F1
    'clf__criterion': ['gini']          # Faster and usually better on tabular
}

# ---------------------------
# Pipeline
# ---------------------------
pipeline = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', rf)
])

# ---------------------------
# RandomizedSearchCV (VERY FAST)
# ---------------------------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=pipeline,
    param_distributions=param_dist,
    n_iter=20,          # FAST – only 20 trials
    scoring='f1',
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("TUNING Hyperparameters (Fast Search)...")
search.fit(X, y)

print("Best Parameters:", search.best_params_)
print("Best F1 CV Score:", search.best_score_)

# ---------------------------
# Predict Test Set
# ---------------------------
final_preds = search.predict(X_test)

submission = pd.DataFrame({
    "patient_id": test_df["patient_id"],
    "has_copd_risk": final_preds
})

submission.to_csv("submission_rf_fast.csv", index=False)
print("Created 'submission_rf_fast.csv'")
