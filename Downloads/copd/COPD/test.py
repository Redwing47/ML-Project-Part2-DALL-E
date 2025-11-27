import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')

# Combine for processing
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['has_copd_risk'] = np.nan
combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. FEATURE ENGINEERING (The Key Improvement)
# BMI: Weight / Height(m)^2
combined['BMI'] = combined['weight_kg'] / ((combined['height_cm'] / 100) ** 2)
# Waist-to-Height Ratio
combined['Waist_Height_Ratio'] = combined['waist_circumference_cm'] / combined['height_cm']
# Pulse Pressure: Systolic - Diastolic
combined['Pulse_Pressure'] = combined['bp_systolic'] - combined['bp_diastolic']
# Cholesterol Ratio
combined['Cholesterol_Ratio'] = combined['total_cholesterol'] / (combined['hdl_cholesterol'] + 1e-5)

# Drop ID
combined = combined.drop(columns=['patient_id'])

# 3. Encoding & Imputation
cat_cols = combined.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    combined[col] = combined[col].fillna('Missing')
    combined[col] = le.fit_transform(combined[col].astype(str))

# Impute (Median is robust and fast)
imp = SimpleImputer(strategy='median')
combined_imputed = pd.DataFrame(imp.fit_transform(combined), columns=combined.columns)

# 4. Split
train_processed = combined_imputed[combined_imputed['is_train'] == 1].copy()
test_processed = combined_imputed[combined_imputed['is_train'] == 0].copy()

X = train_processed.drop(columns=['has_copd_risk', 'is_train'])
y = train_processed['has_copd_risk'].astype(int)
X_test = test_processed.drop(columns=['has_copd_risk', 'is_train'])

# 5. Training (Standard RF performed best)
# Increased estimators to 300 for stability
rf = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf.fit(X, y)

# 6. Prediction
predictions = rf.predict(X_test)

# Save
submission = pd.DataFrame({
    'patient_id': pd.read_csv('test (1).csv')['patient_id'],
    'has_copd_risk': predictions
})
submission.to_csv('submission_optimized.csv', index=False)
print("Optimized submission file created.")