"""Run a single-image inference -> aesthetic score + AI/Real label.

Changes:
- Delay loading of the trained pipeline until first inference call so importing
  this module in a notebook prior to training doesn't fail.
- Use a relaxed import strategy for embed_image so this module works both
  when imported as `src.infer` and when executed directly.
"""
import joblib
import json

try:
    # package import when used as `from src.infer import score_image`
    from .embed_clip import embed_image
except Exception:
    # fallback for direct execution in some environments
    from embed_clip import embed_image

THRESH = 5.0  # tune on validation
_pipe = None


def _load_pipe():
    global _pipe
    if _pipe is None:
        _pipe = joblib.load("models/mlp_aesthetic.joblib")
    return _pipe


def score_image(path: str):
    """Return a dict with aesthetic_score and label ('AI'/'Real')."""
    pipe = _load_pipe()
    s = float(pipe.predict([embed_image(path)])[0])
    return {"aesthetic_score": round(s, 3), "label": ("AI" if s >= THRESH else "Real")}


if __name__ == "__main__":
    import sys
    print(json.dumps(score_image(sys.argv[1]), indent=2))

