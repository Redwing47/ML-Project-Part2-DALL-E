import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, ExtraTreesClassifier, VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')
test_ids = test_df['patient_id']

# 2. Preprocessing
train_df = train_df.drop(columns=['patient_id'])
test_df = test_df.drop(columns=['patient_id'])

# Encode Categorical Columns
categorical_cols = ['sex', 'oral_health_status', 'tartar_presence']
# We combine them briefly to ensure encoding is identical
combined = pd.concat([train_df[categorical_cols], test_df[categorical_cols]], axis=0)
le = LabelEncoder()
for col in categorical_cols:
    combined[col] = le.fit_transform(combined[col])
    train_df[col] = combined.iloc[:len(train_df)][col]
    test_df[col] = combined.iloc[len(train_df):][col]

X = train_df.drop(columns=['has_copd_risk'])
y = train_df['has_copd_risk']

# 3. Imputation (Critical for Ensembles)
# RF and ExtraTrees crash on NaNs, so we fill them with the median value
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)
test_imputed = imputer.transform(test_df)

# 4. Define the "Experts" for our Ensemble
# Model A: The Brain (Gradient Boosting) - usually the most accurate
clf_boost = HistGradientBoostingClassifier(
    max_iter=300,
    learning_rate=0.04,
    max_depth=6,
    random_state=42
)

# Model B: The Wisdom of Crowds (Random Forest)
clf_rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# Model C: The Wildcard (Extra Trees) - adds randomness to prevent overfitting
clf_et = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

# 5. Build the Voting Ensemble
# 'soft' voting means we average the probabilities (confidence), not just the yes/no votes.
# weights=[2, 1, 1] means we trust Boosting twice as much as the others.
voting_model = VotingClassifier(
    estimators=[('boost', clf_boost), ('rf', clf_rf), ('et', clf_et)],
    voting='soft',
    weights=[2, 1, 1] 
)

# 6. Train and Validate
X_train, X_val, y_train, y_val = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

print("Training Ensemble...")
voting_model.fit(X_train, y_train)

val_preds = voting_model.predict(X_val)
print("\nEnsemble Validation Results:")
print(classification_report(y_val, val_preds))
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")

# 7. Create Submission
final_predictions = voting_model.predict(test_imputed)

submission = pd.DataFrame({
    'patient_id': test_ids,
    'has_copd_risk': final_predictions
})

submission.to_csv('submission_ensemble-2.csv', index=False)
print("Submission file 'submission_ensemble.csv' created.")