"""CLIP embedding helpers."""
import torch, numpy as np
from PIL import Image
from pathlib import Path
from transformers import CLIPProcessor, CLIPModel

_device = "cuda" if torch.cuda.is_available() else "cpu"
_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(_device)
_proc  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(path: str) -> np.ndarray:
    img = Image.open(path).convert("RGB")
    inputs = _proc(images=img, return_tensors="pt").to(_device)
    with torch.no_grad():
        z = _model.get_image_features(**inputs)
    z = z / z.norm(dim=-1, keepdim=True)
    return z.squeeze().cpu().numpy()

def embed_dir(in_dir: str, out_npz: str) -> None:
    X, y = [], []
    for lbl in ["real", "ai"]:
        for p in Path(in_dir, lbl).rglob("*.jpg"):
            X.append(embed_image(str(p)))
            y.append(0 if lbl == "real" else 1)
    np.savez_compressed(out_npz, X=np.stack(X), y=np.array(y))
