"""JPEG quality downsampling utility to simulate vendor compression."""
from pathlib import Path
from PIL import Image

def compress_dir(in_dir: str, out_dir: str, qualities=(100, 50, 10, 5)) -> None:
    in_p, out_p = Path(in_dir), Path(out_dir)
    out_p.mkdir(parents=True, exist_ok=True)
    for p in in_p.rglob("*.*"):
        if p.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
            continue
        img = Image.open(p).convert("RGB")
        for q in qualities:
            dest = out_p / f"q{q}" / p.relative_to(in_p)
            dest = dest.with_suffix(".jpg")
            dest.parent.mkdir(parents=True, exist_ok=True)
            img.save(dest, "JPEG", quality=q, optimize=True)
