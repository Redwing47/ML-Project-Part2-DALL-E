import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')

# Combine for consistent preprocessing
train_df['is_train'] = 1
test_df['is_train'] = 0
test_df['has_copd_risk'] = np.nan

combined = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)

# 2. Conservative Feature Engineering
# We only keep the medically proven ratios to reduce noise.
combined['BMI'] = combined['weight_kg'] / ((combined['height_cm'] / 100) ** 2)
combined['Waist_Height_Ratio'] = combined['waist_circumference_cm'] / combined['height_cm']
# Pulse pressure is a strong heart/lung indicator
combined['Pulse_Pressure'] = combined['bp_systolic'] - combined['bp_diastolic']

# Drop ID
combined = combined.drop(columns=['patient_id'])

# 3. Data Preparation
# Identify columns
cat_cols = combined.select_dtypes(include=['object']).columns.tolist()
num_cols = combined.select_dtypes(include=['float64', 'int64']).columns.tolist()
num_cols.remove('has_copd_risk')
num_cols.remove('is_train')

# Fill Missing Values (Conservative approach)
# Categorical -> 'Missing'
combined[cat_cols] = combined[cat_cols].fillna('Missing')
# Numeric -> Median
for col in num_cols:
    combined[col] = combined[col].fillna(combined[col].median())

# Encoding
# We use Ordinal Encoding for Tree models (preserves more info than OneHot for High Cardinality)
oe = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
combined[cat_cols] = oe.fit_transform(combined[cat_cols])

# 4. Split Back
train_processed = combined[combined['is_train'] == 1].copy()
test_processed = combined[combined['is_train'] == 0].copy()

X = train_processed.drop(columns=['has_copd_risk', 'is_train'])
y = train_processed['has_copd_risk'].astype(int)
X_test = test_processed.drop(columns=['has_copd_risk', 'is_train'])

# 5. Define Robust Models
# We increase min_samples_leaf to force the model to look at groups of patients, not individuals.

# Model 1: HistGradientBoosting (Best for tabular data)
# L2 regularization helps prevent overfitting
hgb = HistGradientBoostingClassifier(
    learning_rate=0.05,
    max_iter=300,
    max_leaf_nodes=31,
    min_samples_leaf=20,  # Strict regularization
    l2_regularization=0.5,
    random_state=42
)

# Model 2: ExtraTrees (Less variance than Random Forest)
et = ExtraTreesClassifier(
    n_estimators=500,
    max_depth=15,         # Limit depth
    min_samples_leaf=10,  # Require 10 patients per leaf
    max_features='sqrt',
    bootstrap=False,      # Use whole dataset for splits
    random_state=42,
    n_jobs=-1
)

# Model 3: Random Forest (Classic baseline, strictly regularized)
rf = RandomForestClassifier(
    n_estimators=500,
    max_depth=15,
    min_samples_leaf=10,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# 6. Voting Ensemble
# We give HistGradientBoosting more weight as it's usually the most accurate
ensemble = VotingClassifier(
    estimators=[
        ('hgb', hgb),
        ('et', et),
        ('rf', rf)
    ],
    voting='soft',
    weights=[2, 1, 1], # Double weight to the boosting model
    n_jobs=-1
)

print("Training Ensemble...")
ensemble.fit(X, y)

# 7. Prediction
final_preds = ensemble.predict(X_test)

# 8. Create Submission
submission = pd.DataFrame({
    'patient_id': pd.read_csv('test (1).csv')['patient_id'],
    'has_copd_risk': final_preds.astype(int)
})

submission.to_csv('ensemble-new.csv', index=False)
print("Created 'submission_stable.csv'. This model is tuned to avoid overfitting.")