import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# 1. Load Data
train_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ml_part2_signal_cluster\train.csv')
test_df = pd.read_csv(r'C:\Users\sarth\OneDrive\Desktop\ml_project\ml_part2_signal_cluster\test.csv')

# 2. Preprocessing
# Encode target labels (Group_A, Group_B, etc. -> 0, 1, 2)
le = LabelEncoder()
y = le.fit_transform(train_df['category'])
X = train_df[['signal_strength', 'response_level']]

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features (Important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(test_df[['signal_strength', 'response_level']])

# 3. Model Training
log_reg = LogisticRegression(random_state=42, multi_class='multinomial', max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

# 4. Evaluation
y_pred_val = log_reg.predict(X_val_scaled)
accuracy = accuracy_score(y_val, y_pred_val)

print(f"Validation Accuracy: {accuracy:.4f}")
print("\nClassification Report:")
print(classification_report(y_val, y_pred_val, target_names=le.classes_))

# 5. Prediction on Test Set
y_test_pred_encoded = log_reg.predict(X_test_scaled)
y_test_pred = le.inverse_transform(y_test_pred_encoded)

# Create submission DataFrame
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': y_test_pred
})

# Save to CSV
submission.to_csv('submission-LR.csv', index=False)