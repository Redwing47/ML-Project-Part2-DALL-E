import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')

# Combine for consistent preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['has_copd_risk'] = np.nan
combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Advanced Feature Engineering (The "Secret Sauce")
# BMI
combined['BMI'] = combined['weight_kg'] / ((combined['height_cm'] / 100) ** 2)
# Waist-to-Height Ratio
combined['Waist_Height_Ratio'] = combined['waist_circumference_cm'] / combined['height_cm']
# Pulse Pressure
combined['Pulse_Pressure'] = combined['bp_systolic'] - combined['bp_diastolic']
# Cholesterol Ratio
combined['Cholesterol_Ratio'] = combined['total_cholesterol'] / combined['hdl_cholesterol']

# Drop ID
combined = combined.drop(columns=['patient_id'])

# 3. Categorical Encoding
cat_cols = combined.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    combined[col] = combined[col].astype(str)
    combined[col] = le.fit_transform(combined[col])

# 4. Advanced Imputation (KNN instead of Median)
# KNN requires all data to be numeric, which we just did.
imputer = KNNImputer(n_neighbors=5)
# We exclude the target from imputation to avoid leakage, but need to keep track of columns
features_to_impute = [c for c in combined.columns if c not in ['has_copd_risk', 'is_train']]
combined[features_to_impute] = imputer.fit_transform(combined[features_to_impute])

# 5. Split back
train_processed = combined[combined['is_train'] == 1].copy()
test_processed = combined[combined['is_train'] == 0].copy()

X = train_processed.drop(columns=['has_copd_risk', 'is_train'])
y = train_processed['has_copd_risk'].astype(int)
X_test = test_processed.drop(columns=['has_copd_risk', 'is_train'])

# 6. Improved Random Forest
# Added class_weight='balanced' to handle potential rare cases of COPD
rf = RandomForestClassifier(random_state=42, class_weight='balanced')

# Slightly expanded grid
param_dist = {
    'n_estimators': [200, 300, 500, 800],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'bootstrap': [True, False]
}

search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=30, # Keep it manageable
    cv=5,
    scoring='roc_auc', # 'roc_auc' is often better than 'accuracy' for medical risk
    n_jobs=-1,
    random_state=42,
    verbose=1
)

print("Tuning with improvements...")
search.fit(X, y)

print(f"Best ROC AUC: {search.best_score_:.4f}")

# Predict
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

# Save
submission = pd.DataFrame({
    'patient_id': pd.read_csv('test (1).csv')['patient_id'],
    'has_copd_risk': predictions
})
submission.to_csv('submission_improved.csv', index=False)
print("Done.")