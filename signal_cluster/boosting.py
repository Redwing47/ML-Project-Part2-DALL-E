import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb

# 1. Load Data
# TIP: If you still get an error, paste the full path to the CSV files here
train_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ml_part2_signal_cluster\train.csv')
test_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ml_part2_signal_cluster\test.csv')

# 2. Preprocessing
X = train_df[['signal_strength', 'response_level']]
y = train_df['category']

# Encode target labels (XGBoost requires numeric targets: 0, 1, 2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split data
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# 3. Train XGBoost Model
# We use 'multi:softmax' for multiclass classification
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='multi:softmax',
    num_class=3,  # We have 3 groups: A, B, C
    random_state=42,
    eval_metric='mlogloss'
)

model.fit(X_train, y_train)

# 4. Evaluate
y_pred_val = model.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, y_pred_val):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val, target_names=le.classes_))

# 5. Predictions on Test Set
y_test_pred_encoded = model.predict(test_df[['signal_strength', 'response_level']])
y_test_pred = le.inverse_transform(y_test_pred_encoded)

# Create submission
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': y_test_pred
})

submission.to_csv('submission_xgboost.csv', index=False)
print("Saved submission_xgboost.csv")