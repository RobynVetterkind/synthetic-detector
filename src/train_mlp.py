"""Train a tiny MLP regressor to produce an 'aesthetic' score.

This module exposes a `train` function so notebooks can call it directly
instead of executing top-level script code. The original file-level behavior
is preserved when run as a script.
"""
import numpy as np, joblib
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def train(embeddings_path: str = "embeddings_train.npz", out_path: str = "models/mlp_aesthetic.joblib") -> None:
    """Load embeddings, train the MLP pipeline, and save the model.

    Args:
        embeddings_path: path to the .npz containing X and y
        out_path: path where the trained joblib model should be saved
    """
    D = np.load(embeddings_path)
    X, y = D["X"], D["y"]  # y: 0=real, 1=ai
    # proxy targets; replace later with real aesthetic labels if desired
    t = np.where(y == 1, 5.5, 4.3).astype(float)

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(hidden_layer_sizes=(256, 128), activation="relu",
                              max_iter=200, random_state=42))
    ])
    pipe.fit(X, t)
    joblib.dump(pipe, out_path)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    # preserve original one-shot behavior when executed as a script
    train()

