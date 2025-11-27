import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# 1. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Preprocessing
# Select features
X = train_df[['signal_strength', 'response_level']]
y = train_df['category']
X_test_submission = test_df[['signal_strength', 'response_level']]

# Encode labels (Group_A -> 0, Group_B -> 1, etc.)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split for validation (80% train, 20% validation)
X_train, X_val, y_train, y_val = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Scale features (CRITICAL for SVM performance)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_submission)

# 3. Train SVM Model
# kernel='linear' creates a straight line/plane boundary. 
# You can change to kernel='rbf' for curved boundaries.
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

# 4. Validate
val_predictions = svm_model.predict(X_val_scaled)
print(f"SVM Validation Accuracy: {accuracy_score(y_val, val_predictions):.4f}")

# 5. Final Prediction on Test Data
# (Optional: Retrain on ALL data for best results)
X_full_scaled = scaler.fit_transform(X)
svm_model.fit(X_full_scaled, y_encoded)

test_predictions = svm_model.predict(X_test_scaled)
test_labels = le.inverse_transform(test_predictions)

# 6. Save Submission
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': test_labels
})
submission.to_csv('submission_svm.csv', index=False)
print("Saved submission_svm.csv")