import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# -------------------------
# Reproducibility (optional)
# -------------------------
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -------------------------
# Paths
# -------------------------
train_path = r"C:\Users\sarth\OneDrive\Desktop\ml_project\ml_part2_signal_cluster\train.csv"
test_path = r"C:\Users\sarth\OneDrive\Desktop\ml_project\ml_part2_signal_cluster\test.csv"
out_path = r"C:\Users\sarth\OneDrive\Desktop\ml_project\ml_part2_signal_cluster\nn_submission.csv"

# -------------------------
# Load data
# -------------------------
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Columns in train.csv:", train_df.columns.tolist())
print("Columns in test.csv:", test_df.columns.tolist())

# Expecting:
# train: ['sample_id', 'signal_strength', 'response_level', 'category']
# test:  ['sample_id', 'signal_strength', 'response_level']

# -------------------------
# Features and target
# -------------------------
feature_cols = ["signal_strength", "response_level"]
target_col = "category"

X = train_df[feature_cols].values
y = train_df[target_col].values

# Encode target labels -> integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(le.classes_)

print("Classes:", le.classes_)
print("Number of classes:", num_classes)

# -------------------------
# Train / validation split
# -------------------------
# Use stratify only if every class has at least 2 samples
class_counts = np.bincount(y_encoded)
print("Class counts:", class_counts)

use_stratify = np.min(class_counts) >= 2

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=SEED,
    stratify=y_encoded if use_stratify else None
)

print("Train shape:", X_train.shape, "Val shape:", X_val.shape)

# -------------------------
# Build Neural Network model
# -------------------------
model = Sequential([
    Dense(64, activation="relu", input_shape=(2,)),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64, activation="relu"),
    BatchNormalization(),
    Dropout(0.2),

    Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# -------------------------
# Callbacks (early stopping & LR scheduling)
# -------------------------
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=15,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-5
)

# -------------------------
# Train model
# -------------------------
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=200,
    batch_size=32,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)

# -------------------------
# Evaluate on validation set
# -------------------------
val_probs = model.predict(X_val)
val_pred = np.argmax(val_probs, axis=1)

print("Validation Accuracy:", accuracy_score(y_val, val_pred))
print(classification_report(y_val, val_pred, target_names=le.classes_))

# -------------------------
# Predict on test set
# -------------------------
X_test = test_df[feature_cols].values
test_probs = model.predict(X_test)
test_pred_int = np.argmax(test_probs, axis=1)

# Convert back to original labels
test_pred_labels = le.inverse_transform(test_pred_int)

# -------------------------
# Create submission file
# -------------------------
if "sample_id" not in test_df.columns:
    raise ValueError("Expected 'sample_id' column in test.csv")

submission = pd.DataFrame({
    "sample_id": test_df["sample_id"],
    "category": test_pred_labels
})

submission.to_csv(out_path, index=False)
print(f"Saved submission file as {out_path}")
