import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 1. Load the Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')
submission_sample = pd.read_csv('sample_submission (1).csv')

# 2. Preprocessing
# Store patient_ids for submission and remove from training data
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

# 3. Scaling (Mandatory for SVM)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

# 4. Split Data for Validation
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Train SVM Model
# kernel='rbf' allows the model to fit non-linear data (curves)
svm_model = SVC(kernel='rbf', random_state=42)
svm_model.fit(X_train, y_train)

# 6. Evaluate
val_predictions = svm_model.predict(X_val)
print("Validation Results:")
print(classification_report(y_val, val_predictions))

# Confusion Matrix Visualization
plt.figure(figsize=(6, 5))
cm = confusion_matrix(y_val, val_predictions)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix (Validation Set)')
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.savefig('confusion_matrix_svm.png')
plt.show()

# 7. Create Submission
final_predictions = svm_model.predict(test_scaled)

submission = pd.DataFrame({
    'patient_id': test_ids,
    'has_copd_risk': final_predictions
})

submission.to_csv('submission_svm.csv', index=False)
print("Submission file 'submission_svm.csv' created successfully.")