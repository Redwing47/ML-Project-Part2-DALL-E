import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
# ---------------------------------------------------------
try:
    train_df = pd.read_csv('train (1).csv')
    test_df = pd.read_csv('test (1).csv')
    submission_df = pd.read_csv('sample_submission (1).csv')
except FileNotFoundError:
    print("Error: Files not found. Please upload the datasets.")

# 2. Advanced Feature Engineering
# ---------------------------------------------------------
def create_medical_features(df):
    df_eng = df.copy()
    
    # Body Mass Index (BMI): Key indicator for respiratory load
    # Height is in cm, weight in kg. BMI = kg / m^2
    df_eng['BMI'] = df_eng['weight_kg'] / ((df_eng['height_cm'] / 100) ** 2)
    
    # Pulse Pressure: Indicator of arterial stiffness (Systolic - Diastolic)
    df_eng['Pulse_Pressure'] = df_eng['bp_systolic'] - df_eng['bp_diastolic']
    
    # Mean Arterial Pressure (MAP)
    df_eng['MAP'] = (df_eng['bp_systolic'] + 2 * df_eng['bp_diastolic']) / 3
    
    # Cholesterol Ratios: Better heart/lung health indicators than raw values
    # Avoid division by zero by adding a small epsilon if needed, though cholesterol shouldn't be 0
    df_eng['Cholesterol_Ratio'] = df_eng['total_cholesterol'] / (df_eng['hdl_cholesterol'] + 1e-5)
    df_eng['LDL_HDL_Ratio'] = df_eng['ldl_cholesterol'] / (df_eng['hdl_cholesterol'] + 1e-5)
    
    # Waist-to-Height Ratio: superior to BMI for abdominal obesity
    df_eng['Waist_Height_Ratio'] = df_eng['waist_circumference_cm'] / df_eng['height_cm']
    
    return df_eng

# Apply engineering
print("Generating new medical features...")
train_eng = create_medical_features(train_df)
test_eng = create_medical_features(test_df)

# 3. Preprocessing
# ---------------------------------------------------------
X = train_eng.drop(['has_copd_risk', 'patient_id'], axis=1)
y = train_eng['has_copd_risk']
X_test = test_eng.drop(['patient_id'], axis=1)

# Identify Categorical vs Numerical
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
numerical_cols = X.select_dtypes(include=['number']).columns.tolist()

print(f"Categorical Features: {categorical_cols}")

# Encode Categorical features manually or with LabelEncoder to ensure consistency
# Using LabelEncoder is efficient for Tree-based models
combined = pd.concat([X[categorical_cols], X_test[categorical_cols]], axis=0)
for col in categorical_cols:
    le = LabelEncoder()
    combined[col] = combined[col].astype(str)
    le.fit(combined[col])
    X[col] = le.transform(X[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))

# Imputation strategies
# Random Forest needs no NaNs. HistGradientBoosting handles them, but for Voting, we impute.
imputer = SimpleImputer(strategy='median')
X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_test_imputed = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# 4. Optimized Model Definitions
# ---------------------------------------------------------

# Model A: Random Forest (Bagging)
# Tuned for generalization: more trees, deeper but controlled
rf_optimized = RandomForestClassifier(
    n_estimators=500,        # More trees = stable predictions
    max_depth=15,            # Deep enough to learn complex patterns
    min_samples_leaf=4,      # Prevents overfitting to noise
    max_features='sqrt',     # Standard for classification
    class_weight='balanced', # Crucial for medical datasets (often imbalanced)
    random_state=42,
    n_jobs=-1
)

# Model B: Histogram Gradient Boosting (Boosting)
# Faster and often more accurate than standard GradientBoosting
hgb_optimized = HistGradientBoostingClassifier(
    max_iter=300,            # Equivalent to n_estimators
    learning_rate=0.03,      # Slower learning rate for better accuracy
    max_depth=10,
    l2_regularization=1.0,   # Regularization to prevent overfitting
    early_stopping=True,     # Stop if validation score doesn't improve
    random_state=42
)

# Ensemble: Soft Voting
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf_optimized),
        ('hgb', hgb_optimized)
    ],
    voting='soft',   # Average probabilities
    weights=[1, 1.5] # Give slightly more weight to Boosting (usually more accurate)
)

# 5. Validation and Training
# ---------------------------------------------------------
# Stratified K-Fold is the gold standard for validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\nRunning Cross-Validation (this may take a minute)...")
scores = cross_val_score(ensemble, X_imputed, y, cv=cv, scoring='roc_auc', n_jobs=-1)
print(f"Mean ROC AUC Score: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit on full data
print("Training final model on full dataset...")
ensemble.fit(X_imputed, y)

# 6. Feature Importance Visualization (Bonus)
# ---------------------------------------------------------
# We extract importance from the Random Forest part of the ensemble
feature_importance = ensemble.named_estimators_['rf'].feature_importances_
feat_imp_df = pd.DataFrame({'Feature': X.columns, 'Importance': feature_importance})
feat_imp_df = feat_imp_df.sort_values(by='Importance', ascending=False).head(10)

plt.figure(figsize=(10, 6))
sns.barplot(data=feat_imp_df, x='Importance', y='Feature', palette='viridis')
plt.title('Top 10 Predictors for COPD Risk')
plt.xlabel('Importance (Random Forest)')
plt.tight_layout()
plt.show()

# 7. Submission
# ---------------------------------------------------------
final_preds = ensemble.predict(X_test_imputed)
submission = pd.DataFrame({
    'patient_id': test_df['patient_id'],
    'has_copd_risk': final_preds
})

submission.to_csv('optimized_submission-3.csv', index=False)
print("Optimized submission saved as 'optimized_submission.csv'")