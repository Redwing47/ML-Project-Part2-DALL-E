import pandas as pd
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score

# Load Data
train_df = pd.read_csv('c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\train (1).csv')
test_df = pd.read_csv('c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\test (1).csv')
sample_sub = pd.read_csv('c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\sample_submission (1).csv')

X = train_df.drop(columns=["has_copd_risk", "patient_id"])
y = train_df["has_copd_risk"].astype(int)

X_test = test_df.drop(columns=["patient_id"])

# Identify column types
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# Preprocessing Pipeline
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

# Random Forest
rf = RandomForestClassifier(random_state=42, n_jobs=-1)

# Better hyperparameter search space
param_dist = {
    'clf__n_estimators': [200, 300, 400, 600],
    'clf__max_depth': [8, 12, 16, None],
    'clf__min_samples_split': [2, 5, 10],
    'clf__min_samples_leaf': [1, 2, 3],
    'clf__bootstrap': [True],
    'clf__criterion': ['gini', 'entropy']
}

# Final pipeline
model = Pipeline(steps=[
    ('preprocess', preprocessor),
    ('clf', rf)
])

# Randomized Search using F1 score
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=model,
    param_distributions=param_dist,
    n_iter=40,
    scoring='f1',         # IMPORTANT — CV uses F1 (competition metric)
    cv=cv,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

print("Tuning Hyperparameters...")
search.fit(X, y)

print("Best Parameters:", search.best_params_)
print("Best F1 CV Score:", search.best_score_)

# Predict Test Set
final_preds = search.predict(X_test)

# Create Submission File
submission = pd.DataFrame({
    "patient_id": test_df["patient_id"],
    "has_copd_risk": final_preds
})

submission.to_csv("submission_rf-6.csv", index=False)
print("Created 'submission_rf-6.csv'")
