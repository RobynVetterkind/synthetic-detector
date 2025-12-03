"""
prepare_real_images.py

Utility script for the Synthetic Image Detector project.

Expected workflow:
- The user downloads raw real-image datasets manually from sources like Kaggle,
  Wikimedia Commons, or other open datasets.
- They unzip/place those images into:
    data_raw/cars/
    data_raw/buildings/
    data_raw/medical/
- This script then:
    * samples a target number of images per category,
    * converts them to RGB,
    * resizes them to 512x512,
    * saves them into data/real/ with standardized filenames:
        cars_0000.jpg, buildings_0000.jpg, medical_0000.jpg, etc.

This prepares a clean "real" dataset ready for CLIP embeddings and modeling.
"""

from pathlib import Path
from PIL import Image
import random

# Where your raw datasets live
SOURCE_DIRS = {
    "cars": Path("data_raw/cars"),
    "buildings": Path("data_raw/buildings"),
    "medical": Path("data_raw/medical"),
}

# How many you want from each
TARGET_COUNTS = {
    "cars": 400,
    "buildings": 300,
    "medical": 115,  # total ~815
}

TARGET_DIR = Path("data/real")
TARGET_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (512, 512)
VALID_EXT = {".jpg", ".jpeg", ".png"}


def curate_category(category: str, src_dir: Path, target_n: int) -> int:
    """Sample, resize, and save images for a single category."""
    if not src_dir.exists():
        print(f"[WARN] Source directory does not exist for {category}: {src_dir}")
        return 0

    files = [p for p in src_dir.rglob("*") if p.suffix.lower() in VALID_EXT]
    if not files:
        print(f"[WARN] No valid image files found for {category} in {src_dir}")
        return 0

    random.shuffle(files)
    target_n = min(target_n, len(files))
    print(f"{category}: using {target_n} images from {len(files)} available")

    saved = 0
    for i, img_path in enumerate(files[:target_n]):
        try:
            img = Image.open(img_path).convert("RGB")
            img = img.resize(IMG_SIZE)
            out_path = TARGET_DIR / f"{category}_{i:04d}.jpg"
            img.save(out_path, "JPEG", quality=95)
            saved += 1
        except Exception as e:
            print(f"[SKIP] {img_path} due to error: {e}")

    return saved


def main():
    total_saved = 0
    for cat, src in SOURCE_DIRS.items():
        n_target = TARGET_COUNTS.get(cat, 0)
        saved = curate_category(cat, src, n_target)
        total_saved += saved

    print("\n=== Curation Summary ===")
    for cat in SOURCE_DIRS.keys():
        count = len(list(TARGET_DIR.glob(f"{cat}_*.jpg")))
        print(f"{cat}: {count} images in {TARGET_DIR}")
    print(f"Total curated images: {total_saved}")


if __name__ == "__main__":
    main()
