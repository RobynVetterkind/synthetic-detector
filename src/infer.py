"""Run a single-image inference -> aesthetic score + AI/Real label."""
import joblib, json
from embed_clip import embed_image

THRESH = 5.0  # tune on validation
_pipe = joblib.load("models/mlp_aesthetic.joblib")

def score_image(path: str):
    s = float(_pipe.predict([embed_image(path)])[0])
    return {"aesthetic_score": round(s, 3), "label": ("AI" if s >= THRESH else "Real")}

if __name__ == "__main__":
    import sys
    print(json.dumps(score_image(sys.argv[1]), indent=2))
