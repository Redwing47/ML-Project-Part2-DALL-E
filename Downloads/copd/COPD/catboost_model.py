import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
import numpy as np

# load data
train = pd.read_csv("c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\train (1).csv")
test = pd.read_csv("c:\\Users\\LEGION\\OneDrive\\Desktop\\ML2.1\\test (1).csv")

# define features and target
features = [col for col in train.columns if col not in ['target', 'id']]
target = 'target'

# create a pool
train_pool = Pool(train[features], train[target])

# define the model
model = CatBoostClassifier(
    iterations=100,
    learning_rate=0.1,
    depth=6,
    verbose=True
)

# train the model
model.fit(train_pool)

# make predictions
test_predictions = model.predict(test[features])

# evaluate the model
f1 = f1_score(test[target], test_predictions)
print(f"F1 score: {f1}")