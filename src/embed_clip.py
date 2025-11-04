"""CLIP embedding helpers.

Notes/changes:
- Model loading is deferred until the first call to avoid heavy network/torch ops at import time
  (helps notebooks import the module without immediately attempting to download/initialize CLIP).
"""
import numpy as np
from PIL import Image
from pathlib import Path
import torch

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = None
_proc = None


def _ensure_model():
    """Load CLIP model and processor on first use.

    This avoids downloading/loading the model during import which can fail in some
    notebook environments or when running lint-only operations.
    """
    global _model, _proc
    if _model is None or _proc is None:
        # local import to keep import-time lightweight
        from transformers import CLIPProcessor, CLIPModel

        _model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
        _proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")


def embed_image(path: str) -> np.ndarray:
    """Return a normalized CLIP image embedding for a single image path."""
    _ensure_model()
    img = Image.open(path).convert("RGB")
    inputs = _proc(images=img, return_tensors="pt").to(_device)
    with torch.no_grad():
        z = _model.get_image_features(**inputs)
    z = z / z.norm(dim=-1, keepdim=True)
    return z.squeeze().cpu().numpy()


def embed_dir(in_dir: str, out_npz: str) -> None:
    """Embed all JPG images under in_dir/real and in_dir/ai and save to out_npz.

    The function will iterate over `real` and `ai` subfolders within `in_dir` and
    save a compressed npz with keys `X` and `y` where `y` is 0 for real, 1 for ai.
    """
    X, y = [], []
    for lbl in ["real", "ai"]:
        for p in Path(in_dir, lbl).rglob("*.jpg"):
            X.append(embed_image(str(p)))
            y.append(0 if lbl == "real" else 1)
    if len(X) == 0:
        raise RuntimeError(f"No .jpg images found under {in_dir}/{{real,ai}} to embed")
    np.savez_compressed(out_npz, X=np.stack(X), y=np.array(y))

