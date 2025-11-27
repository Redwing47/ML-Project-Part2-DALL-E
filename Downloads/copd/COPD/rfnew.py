import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')
submission_sample = pd.read_csv('sample_submission (1).csv')

# 2. Preprocessing
# Store patient_ids for submission
test_ids = test_df['patient_id']
train_df = train_df.drop(columns=['patient_id'])
test_df = test_df.drop(columns=['patient_id'])

# Encode Categorical Columns
categorical_cols = ['sex', 'oral_health_status', 'tartar_presence']
le = LabelEncoder()

for col in categorical_cols:
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Separate Features (X) and Target (y)
X = train_df.drop(columns=['has_copd_risk'])
y = train_df['has_copd_risk']

# 3. Split Data (No Scaling needed for Random Forest)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train Random Forest Model
# n_estimators=100 creates 100 decision trees
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# 5. Evaluate
val_predictions = rf_model.predict(X_val)
print("Validation Results:")
print(classification_report(y_val, val_predictions))

# 6. Feature Importance Visualization
# This shows which variables matter most for the diagnosis
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.title("Top Factors Predicting COPD Risk")
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.tight_layout()
plt.show()

# 7. Create Submission
final_predictions = rf_model.predict(test_df)

submission = pd.DataFrame({
    'patient_id': test_ids,
    'has_copd_risk': final_predictions
})

submission.to_csv('submission_rf_new.csv', index=False)
print("Submission file 'submission_rf.csv' created successfully.")