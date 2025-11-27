svm_with_knn_stacking.pyimport pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# 1. Load Data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X = train_df[['signal_strength', 'response_level']]
y = train_df['category']
X_test = test_df[['signal_strength', 'response_level']]

# 2. Pipeline with Scaling
# KNN strictly requires scaling. Pipeline ensures it's done correctly.
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
])

# 3. Hyperparameter Tuning (Grid Search)
param_grid = {
    'knn__n_neighbors': range(1, 31),        # Test K from 1 to 30
    'knn__weights': ['uniform', 'distance'], # uniform vs distance weighting
    'knn__metric': ['euclidean', 'manhattan'] # distance metric
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print(f"Best Params: {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_:.4f}")

# 4. Predict and Save
best_model = grid_search.best_estimator_
predictions = best_model.predict(X_test)

submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': predictions
})
submission.to_csv('submission_knn_tuned.csv', index=False)