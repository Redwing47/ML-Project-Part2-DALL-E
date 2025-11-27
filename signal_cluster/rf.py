import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# 1. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df[['signal_strength', 'response_level']]
y = train_df['category']
X_test = test_df[['signal_strength', 'response_level']]

# 2. Scaling (Optional for RF but good practice if comparing with distance-based models)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 3. Hyperparameter Tuning for Random Forest
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_scaled, y)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.4f}")

# 4. Train Final Model and Predict
best_rf = grid_search.best_estimator_
best_rf.fit(X_scaled, y)
predictions = best_rf.predict(X_test_scaled)

# 5. Save Submission
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': predictions
})
submission.to_csv('submission_rf.csv', index=False)