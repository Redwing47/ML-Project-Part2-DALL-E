import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression

# 1. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df[['signal_strength', 'response_level']]
y = train_df['category']
X_test = test_df[['signal_strength', 'response_level']]

# 2. Preprocessing
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale Features (Crucial for SVM & KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. Define Stacking Ensemble
# We combine variations of KNN and SVM to give the meta-learner diverse inputs
estimators = [
    ('knn_5', KNeighborsClassifier(n_neighbors=5)),
    ('knn_15', KNeighborsClassifier(n_neighbors=15)),
    ('svm_rbf', SVC(kernel='rbf', probability=True, random_state=42)),
    ('svm_lin', SVC(kernel='linear', probability=True, random_state=42))
]

# The StackingClassifier automatically uses Cross-Validation to train the meta-learner
stacking_model = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(),
    stack_method='predict_proba', # Use probability outputs for better information
    n_jobs=-1
)

# 4. Train on Full Data
stacking_model.fit(X_scaled, y_encoded)

# 5. Predict
final_preds_encoded = stacking_model.predict(X_test_scaled)
final_preds_labels = le.inverse_transform(final_preds_encoded)

# 6. Save
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': final_preds_labels
})
submission.to_csv('submission_stacking.csv', index=False)
print("Submission saved.")