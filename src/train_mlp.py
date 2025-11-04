"""Train a tiny MLP regressor to produce an 'aesthetic' score."""
import numpy as np, joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

D = np.load("embeddings_train.npz")
X, y = D["X"], D["y"]  # y: 0=real, 1=ai
# proxy targets; replace later with real aesthetic labels if desired
t = np.where(y == 1, 5.5, 4.3).astype(float)

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("mlp", MLPRegressor(hidden_layer_sizes=(256,128), activation="relu",
                         max_iter=200, random_state=42))
])
pipe.fit(X, t)
joblib.dump(pipe, "models/mlp_aesthetic.joblib")
print("Saved models/mlp_aesthetic.joblib")
