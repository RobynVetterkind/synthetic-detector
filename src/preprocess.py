"""Image preprocessing helpers.

This module provides a simple helper to open an image, convert it to RGB,
resize it (default 224x224), and save it as a JPEG with adjustable quality.

The function creates parent directories for the output path if they do not
exist. It is intentionally small and dependency-light to be usable in
notebooks and small scripts.
"""
from pathlib import Path
from typing import Tuple, Union

from PIL import Image


def preprocess_image(
    in_path: Union[str, Path],
    out_path: Union[str, Path],
    size: Tuple[int, int] = (224, 224),
    quality: int = 95,
) -> Path:
    """Open an image, convert to RGB, resize, and save as JPEG.

    Args:
        in_path: Path to the input image file.
        out_path: Path where the processed JPEG will be written.
        size: Desired output size as (width, height). Defaults to (224, 224).
        quality: JPEG quality (1-100). Higher means better quality. Defaults to 95.

    Returns:
        The Path to the written output file.
    """
    in_p = Path(in_path)
    out_p = Path(out_path)

    # Ensure the input exists — let PIL raise a FileNotFoundError if not.
    with Image.open(in_p) as im:
        im = im.convert("RGB")
        # Use a high-quality resampling filter
        im = im.resize(size, Image.LANCZOS)

        out_p.parent.mkdir(parents=True, exist_ok=True)
        # Always save as JPEG (common for downstream CLIP ingestion)
        im.save(out_p, format="JPEG", quality=int(quality), optimize=True)

    return out_p


__all__ = ["preprocess_image"]
