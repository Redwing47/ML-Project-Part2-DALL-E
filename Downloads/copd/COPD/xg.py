import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
# We use HistGradientBoostingClassifier as the robust alternative to XGBoost
from sklearn.ensemble import HistGradientBoostingClassifier

# 1. Load Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')

# 2. Preprocessing
test_ids = test_df['patient_id']
train_df = train_df.drop(columns=['patient_id'])
test_df = test_df.drop(columns=['patient_id'])

# Encode Categorical Columns
categorical_cols = ['sex', 'oral_health_status', 'tartar_presence']
le = LabelEncoder()

for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

X = train_df.drop(columns=['has_copd_risk'])
y = train_df['has_copd_risk']

# 3. Split Data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Model (Boosting)
# This mimics the behavior of XGBoost/LightGBM
model = HistGradientBoostingClassifier(
    max_iter=200,        # Similar to n_estimators
    learning_rate=0.05,  # Lower learning rate often yields better generalization
    max_depth=5,
    random_state=42
)

print("Training Boosting Model...")
model.fit(X_train, y_train)

# 5. Evaluate
val_predictions = model.predict(X_val)
print("Validation Results:")
print(classification_report(y_val, val_predictions))

# 6. Create Submission
final_predictions = model.predict(test_df)

submission = pd.DataFrame({
    'patient_id': test_ids,
    'has_copd_risk': final_predictions
})

submission.to_csv('submission_xgboost.csv', index=False)
print("Submission file 'submission_xgboost.csv' created successfully.")