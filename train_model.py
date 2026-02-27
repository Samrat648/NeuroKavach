import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

# Fake training data
X = np.random.rand(500, 3)
y = np.where(X[:, 0] > 0.7, 1, 0)

model = RandomForestClassifier()
model.fit(X, y)

joblib.dump(model, "model.pkl")

print("Model trained and saved.")
