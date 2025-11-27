import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the Data
train_df = pd.read_csv('train (1).csv')
test_df = pd.read_csv('test (1).csv')
submission_sample = pd.read_csv('sample_submission (1).csv')

# 2. Preprocessing
# Store patient_ids for the final submission and drop them from training
test_ids = test_df['patient_id']
train_df = train_df.drop(columns=['patient_id'])
test_df = test_df.drop(columns=['patient_id'])

# Identify Categorical Columns (Text data)
categorical_cols = ['sex', 'oral_health_status', 'tartar_presence']

# Encode Categorical Columns (Convert Text to Numbers)
le = LabelEncoder()
for col in categorical_cols:
    # Fit on train and transform both to keep consistency
    train_df[col] = le.fit_transform(train_df[col])
    test_df[col] = le.transform(test_df[col])

# Separate Features (X) and Target (y)
X = train_df.drop(columns=['has_copd_risk'])
y = train_df['has_copd_risk']

# 3. Scaling (Crucial for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_scaled = scaler.transform(test_df)

# 4. Split Training Data for Validation
# We split the training data to test our model's accuracy before predicting on the real test set
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 5. Find the Best 'K' Value
# We check K values from 1 to 20 to see which gives the lowest error
error_rate = []
for i in range(1, 21):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_val)
    error_rate.append(np.mean(pred_i != y_val))

# Pick the K with the minimum error (or default to 9 if uncertain)
best_k = error_rate.index(min(error_rate)) + 1
print(f"Optimal K found: {best_k}")

# 6. Train Final Model
knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(X_train, y_train)

# Evaluate on Validation Set
val_predictions = knn.predict(X_val)
print("\nValidation Results:")
print(classification_report(y_val, val_predictions))

# 7. Predict on Actual Test Data
final_predictions = knn.predict(test_scaled)

# 8. Create Submission File
submission = pd.DataFrame({
    'patient_id': test_ids,
    'has_copd_risk': final_predictions
})

# Save to CSV
submission.to_csv('submission-knn.csv', index=False)
print("Submission file 'submission.csv' created successfully.")