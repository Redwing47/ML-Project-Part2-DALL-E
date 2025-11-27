import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')
sample_sub = pd.read_csv('sample_submission (1).csv')

# 2. Data Preprocessing

# Combine for consistent preprocessing (handling categoricals)
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['has_copd_risk'] = np.nan # Placeholder for target

combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# Drop ID columns usually not useful for prediction
combined = combined.drop(columns=['patient_id'])

# Handle Missing Values (Numeric)
numeric_cols = combined.select_dtypes(include=['float64', 'int64']).columns
# Exclude the target and the marker 'is_train' from imputation if needed, 
# but SimpleImputer ignores NaNs in target if we are careful.
# Actually simpler to impute strictly features.

features = [c for c in combined.columns if c not in ['has_copd_risk', 'is_train']]

# Impute numeric features with Median (robust to outliers)
num_imputer = SimpleImputer(strategy='median')
combined[features] = combined[features].copy() # avoid copy warning

# Identify Categorical Columns
cat_cols = combined[features].select_dtypes(include=['object']).columns

# Encode Categorical Variables
# For 'sex' (M/F) and 'Y/N' columns
le = LabelEncoder()
for col in cat_cols:
    combined[col] = combined[col].fillna(combined[col].mode()[0]) # Fill categorical NaNs with mode
    combined[col] = le.fit_transform(combined[col].astype(str))

# 3. Split back into Train and Test
train_processed = combined[combined['is_train'] == 1].copy()
test_processed = combined[combined['is_train'] == 0].copy()

# Define X and y
X = train_processed.drop(columns=['has_copd_risk', 'is_train'])
y = train_processed['has_copd_risk'].astype(int)
X_test = test_processed.drop(columns=['has_copd_risk', 'is_train'])

# 4. Model Training & Hyperparameter Tuning
# Random Forest is robust, but tuning improves score significantly.

rf = RandomForestClassifier(random_state=42)

# Define Hyperparameter Grid
param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False],
    'criterion': ['gini', 'entropy']
}

# Use RandomizedSearchCV for efficiency and performance
search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,               # Number of parameter settings that are sampled
    cv=5,                    # 5-fold Cross validation
    verbose=1,
    random_state=42,
    n_jobs=-1,               # Use all available cores
    scoring='accuracy'       # Optimizing for Accuracy (standard for classification)
)

print("Tuning Hyperparameters...")
search.fit(X, y)

print(f"Best Parameters: {search.best_params_}")
print(f"Best CV Score: {search.best_score_:.4f}")

# 5. Final Prediction
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

# 6. Create Submission File
submission = pd.DataFrame({
    'patient_id': pd.read_csv('test (1).csv')['patient_id'],
    'has_copd_risk': predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file 'submission.csv' created successfully.")