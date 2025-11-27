import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load Data
train_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ML2.1\train (1).csv')
test_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ML2.1\test (1).csv')

# 2. Preprocessing
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

# 3. Model Definition (Pipeline)
# Scaling is critical for Logistic Regression
model = Pipeline([
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000))
])

# 4. Evaluation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=cv, scoring='f1')
print(f"Logistic Regression F1 Score: {scores.mean():.4f}")

# 5. Prediction
model.fit(X, y)
predictions = model.predict(X_test)

# 6. Save Submission
submission = pd.DataFrame({
    'patient_id': test_df['patient_id'],
    'has_copd_risk': predictions
})
submission.to_csv('submission_logistic.csv', index=False)