import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Load the datasets
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# 2. Prepare feature matrix (X) and target vector (y)
# We use the two signal measurements as features
X = train_df[['signal_strength', 'response_level']]
y = train_df['category']
X_test = test_df[['signal_strength', 'response_level']]

# 3. Scale the features
# KNN is distance-based, so scaling is crucial for good performance
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_test_scaled = scaler.transform(X_test)

# 4. Split training data for validation
# This helps us estimate model performance before predicting on the test set
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Hyperparameter Tuning using GridSearchCV
# We search for the best 'n_neighbors' (k) from 1 to 30
knn = KNeighborsClassifier()
param_grid = {'n_neighbors': np.arange(1, 31)}

grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_split, y_train_split)

print(f"Best K found: {grid_search.best_params_['n_neighbors']}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# 6. Evaluate on the validation set
best_knn = grid_search.best_estimator_
val_predictions = best_knn.predict(X_val_split)
print("\nValidation Accuracy:", accuracy_score(y_val_split, val_predictions))
print("\nClassification Report:\n", classification_report(y_val_split, val_predictions))

# 7. Retrain on the FULL training dataset
# Now that we know the best K, we use all available training data
final_knn = KNeighborsClassifier(n_neighbors=grid_search.best_params_['n_neighbors'])
final_knn.fit(X_scaled, y)

# 8. Predict on the test set
test_predictions = final_knn.predict(X_test_scaled)

# 9. Create and save the submission file
submission = pd.DataFrame({
    'sample_id': test_df['sample_id'],
    'category': test_predictions
})

submission.to_csv('submission.csv', index=False)
print("Submission file created: submission.csv")