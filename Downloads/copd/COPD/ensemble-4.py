import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')
sample_sub = pd.read_csv('sample_submission (1).csv')

# 2. Advanced Feature Engineering (The biggest ROI for accuracy)
def create_medical_features(df):
    df = df.copy()
    
    # Body Mass Index (BMI)
    # Height is usually in cm, convert to meters
    df['BMI'] = df['weight_kg'] / ((df['height_cm'] / 100) ** 2)
    
    # Waist-to-Height Ratio (Indicator of central obesity)
    df['Waist_Height_Ratio'] = df['waist_circumference_cm'] / df['height_cm']
    
    # Pulse Pressure (Systolic - Diastolic) - heart health indicator
    df['Pulse_Pressure'] = df['bp_systolic'] - df['bp_diastolic']
    
    # Cholesterol Ratios
    # Avoid division by zero by adding a tiny epsilon if needed, 
    # though medical data implies non-zero cholesterol usually.
    df['Chol_HDL_Ratio'] = df['total_cholesterol'] / df['hdl_cholesterol']
    df['LDL_HDL_Ratio'] = df['ldl_cholesterol'] / df['hdl_cholesterol']
    
    # Liver Enzyme Ratios
    df['AST_ALT_Ratio'] = df['ast_enzyme_level'] / df['alt_enzyme_level']
    
    return df

# Apply feature engineering BEFORE splitting
train_df = create_medical_features(train_df)
test_df = create_medical_features(test_df)

# 3. Preparation without Leakage

# Separate Target and IDs
X = train_df.drop(columns=['patient_id', 'has_copd_risk'])
y = train_df['has_copd_risk']
X_test = test_df.drop(columns=['patient_id'])

# Identify Columns
numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

# --- PREPROCESSING PIPELINE (Manual for clarity) ---

# A. Imputation (Fit on Train, Transform Test)
# Using median is robust for medical data
imputer = SimpleImputer(strategy='median')
X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
X_test[numeric_cols] = imputer.transform(X_test[numeric_cols])

# B. Encoding
# We use OrdinalEncoder for features (handles unknown categories better if configured)
# or LabelEncoder loop. Since we have standard 'Y/N' and 'M/F', mapping is safer.

# Manual robust mapping to ensure consistency
binary_map = {'N': 0, 'Y': 1, 'M': 0, 'F': 1}

for col in cat_cols:
    # Map known binary columns if they match the pattern, otherwise LabelEncode
    if set(X[col].dropna().unique()).issubset({'Y', 'N', 'M', 'F', 'nan'}):
        X[col] = X[col].map(binary_map)
        X_test[col] = X_test[col].map(binary_map)
    else:
        # Fallback for other categorical columns
        le = LabelEncoder()
        # Handle new categories in test by fitting on combined unique values 
        # (This is acceptable for Label Encoding generally, or use 'unknown')
        # However, strictly:
        X[col] = X[col].astype(str)
        X_test[col] = X_test[col].astype(str)
        le.fit(pd.concat([X[col], X_test[col]])) 
        X[col] = le.transform(X[col])
        X_test[col] = le.transform(X_test[col])

# Fill any remaining NaNs created by mapping (if any)
X = X.fillna(-1)
X_test = X_test.fillna(-1)

# 4. Model Training & Tuning

# Added class_weight='balanced' to handle case where fewer people have COPD than not
rf = RandomForestClassifier(
    random_state=42, 
    class_weight='balanced', 
    n_jobs=-1
)

param_dist = {
    'n_estimators': [300, 500, 800], # Generally more is better for RF, just slower
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'], # Important parameter
    'bootstrap': [True]
}

# StratifiedKFold ensures each fold has same proportion of COPD cases
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=20, # Reduced iter to keep it faster, increase if you have time
    cv=cv,
    verbose=1,
    random_state=42,
    n_jobs=-1,
    scoring='accuracy' 
)

print("Tuning Hyperparameters...")
search.fit(X, y)

print(f"Best Parameters: {search.best_params_}")
print(f"Best CV Score: {search.best_score_:.4f}")

# 5. Final Prediction
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

# 6. Submission
submission = pd.DataFrame({
    'patient_id': test_df['patient_id'],
    'has_copd_risk': predictions
})

submission.to_csv('submission_optimized-4.csv', index=False)
print("Submission file 'submission_optimized-4.csv' created successfully.")