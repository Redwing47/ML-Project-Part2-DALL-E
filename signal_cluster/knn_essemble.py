import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# 1. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df[['signal_strength', 'response_level']]
y = train_df['category']
X_test = test_df[['signal_strength', 'response_level']]

# 2. Preprocessing
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale features (Critical for KNN)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. Create Ensemble of KNNs
# We use different 'k' values to capture both local (small k) and global (large k) patterns
knn3 = KNeighborsClassifier(n_neighbors=3)
knn5 = KNeighborsClassifier(n_neighbors=5)
knn7 = KNeighborsClassifier(n_neighbors=7)
knn9 = KNeighborsClassifier(n_neighbors=9)

# Soft Voting: Averages the probabilities from each model (better than hard voting)
ensemble_model = VotingClassifier(
    estimators=[
        ('k3', knn3), 
        ('k5', knn5), 
        ('k7', knn7), 
        ('k9', knn9)
    ],
    voting='soft'
)

# 4. Train
ensemble_model.fit(X_scaled, y_encoded)

# 5. Predict
final_preds_encoded = ensemble_model.predict(X_test_scaled)
final_preds_labels = le.inverse_transform(final_preds_encoded)

# 6. Save
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': final_preds_labels
})
submission.to_csv('submission_knn_ensemble.csv', index=False)
print("Saved submission_knn_ensemble.csv")