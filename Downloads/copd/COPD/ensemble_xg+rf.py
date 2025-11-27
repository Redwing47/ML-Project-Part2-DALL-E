import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, HistGradientBoostingClassifier

# ---------------------------------------------------------
# 1. Load Data
# ---------------------------------------------------------
train_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ML2.1\train (1).csv')
test_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ML2.1\test (1).csv')

# ---------------------------------------------------------
# 2. Preprocessing
# ---------------------------------------------------------
def preprocess(df):
    df = df.copy()
    # Drop identifier and constant columns
    cols_to_drop = ['patient_id', 'oral_health_status']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Map binary categoricals
    if df['sex'].dtype == 'object':
        df['sex'] = df['sex'].map({'M': 0, 'F': 1})
    if df['tartar_presence'].dtype == 'object':
        df['tartar_presence'] = df['tartar_presence'].map({'N': 0, 'Y': 1})
        
    return df

X = preprocess(train_df)
y = X.pop('has_copd_risk')
X_test = preprocess(test_df)

# ---------------------------------------------------------
# 3. Model Definition
# ---------------------------------------------------------

# Random Forest (Bagging)
rf_model = RandomForestClassifier(
    n_estimators=200,
    max_depth=10,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

# Gradient Boosting (Boosting)
# Note: You can replace this with XGBClassifier() if xgboost is installed.
# We use HistGradientBoostingClassifier as it handles larger data/NaNs efficiently similar to XGB.
gb_model = HistGradientBoostingClassifier(
    max_iter=200,
    max_depth=6,
    learning_rate=0.05,
    class_weight='balanced', # Available in sklearn >= 1.2
    random_state=42
)

# Ensemble (Soft Voting)
ensemble_model = VotingClassifier(
    estimators=[('rf', rf_model), ('gb', gb_model)],
    voting='soft'
)

# ---------------------------------------------------------
# 4. Training and Prediction
# ---------------------------------------------------------
print("Training Ensemble Model...")
ensemble_model.fit(X, y)

print("Predicting on Test Set...")
predictions = ensemble_model.predict(X_test)

# ---------------------------------------------------------
# 5. Create Submission
# ---------------------------------------------------------
submission = pd.DataFrame({
    'patient_id': test_df['patient_id'],
    'has_copd_risk': predictions
})

submission.to_csv('submission_ensemble_xg+rf.csv', index=False)
print("Done! Saved to submission_ensemble_xg+rf.csv")