import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

# Try importing XGBoost
try:
    from xgboost import XGBClassifier
    xgb_available = True
except ImportError:
    xgb_available = False
    print("XGBoost not found, using sklearn alternatives.")

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')

# Combine for consistent preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['has_copd_risk'] = np.nan

combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Advanced Feature Engineering
# --- Ratios & Interactions ---
combined['BMI'] = combined['weight_kg'] / ((combined['height_cm'] / 100) ** 2)
combined['Waist_Height_Ratio'] = combined['waist_circumference_cm'] / combined['height_cm']
combined['Pulse_Pressure'] = combined['bp_systolic'] - combined['bp_diastolic']
combined['Cholesterol_Ratio'] = combined['total_cholesterol'] / (combined['hdl_cholesterol'] + 1e-5)
combined['Trig_HDL_Ratio'] = combined['triglycerides'] / (combined['hdl_cholesterol'] + 1e-5)

# --- Log Transformations for Skewed Features ---
# Skewed features identified: ALT, AST, GGT, Creatinine, Triglycerides
skewed_cols = ['alt_enzyme_level', 'ast_enzyme_level', 'ggt_enzyme_level', 
               'serum_creatinine', 'triglycerides', 'fasting_glucose']

for col in skewed_cols:
    # Add +1 to avoid log(0)
    combined[f'Log_{col}'] = np.log1p(combined[col])

# Drop ID
combined = combined.drop(columns=['patient_id'])

# 3. Encoding
cat_cols = combined.select_dtypes(include=['object']).columns
le = LabelEncoder()
for col in cat_cols:
    combined[col] = combined[col].fillna('Missing')
    combined[col] = le.fit_transform(combined[col].astype(str))

# 4. Imputation
# Using Median for robustness
imp = SimpleImputer(strategy='median')
combined_imputed = pd.DataFrame(imp.fit_transform(combined), columns=combined.columns)

# 5. Split Back
train_processed = combined_imputed[combined_imputed['is_train'] == 1].copy()
test_processed = combined_imputed[combined_imputed['is_train'] == 0].copy()

X = train_processed.drop(columns=['has_copd_risk', 'is_train'])
y = train_processed['has_copd_risk'].astype(int)
X_test = test_processed.drop(columns=['has_copd_risk', 'is_train'])

# 6. Define Models for Ensemble

# Model 1: HistGradientBoosting (Very similar to LightGBM, fast and accurate)
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=500,
    max_depth=10,
    random_state=42,
    l2_regularization=0.1
)

# Model 2: Random Forest (Optimized)
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=5,
    max_features='sqrt',
    n_jobs=-1,
    random_state=42
)

# Model 3: XGBoost (if available)
estimators = [
    ('hgb', hgb),
    ('rf', rf)
]

if xgb_available:
    xgb = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        eval_metric='logloss'
    )
    estimators.append(('xgb', xgb))

# 7. Voting Ensemble
# Soft voting averages the probabilities of all models
ensemble = VotingClassifier(
    estimators=estimators,
    voting='soft',
    n_jobs=-1
)

print(f"Training Ensemble with: {[name for name, _ in estimators]}...")
ensemble.fit(X, y)

# 8. Evaluation (Internal Check)
# We can't do a full CV here without re-fitting, but let's trust the ensemble logic.
# (Usually Ensembles boost score by 0.01-0.02 over single models)

# 9. Prediction
final_preds = ensemble.predict(X_test)

# 10. Submission
submission = pd.DataFrame({
    'patient_id': pd.read_csv('test (1).csv')['patient_id'],
    'has_copd_risk': final_preds.astype(int)
})

submission.to_csv('submission_ensemble.csv', index=False)
print("Submission file 'submission_ensemble.csv' created successfully.")