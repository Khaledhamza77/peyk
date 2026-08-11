"""Q1's blur-detection trigger (docs-personal/surya/improvement.md): variance of a 3x3
Laplacian-filtered image. Chosen over three other sharpness metrics tested (tenengrad,
brenner, modified_laplacian) for giving the widest relative separation between confirmed
fail/pass cases (3.39x vs. 2.60x/1.69x/1.54x) — see that doc's Q1 for the full evidence and
conceptual justification. Ported from z_surya_exploration/measure_noise.py — only this one
metric, the other three were comparison-only during metric selection, never part of the
actual decision logic."""
import numpy as np
from PIL import Image, ImageFilter

LAPLACIAN_KERNEL = ImageFilter.Kernel((3, 3), [0, 1, 0, 1, -4, 1, 0, 1, 0], scale=1)


def laplacian_variance(image: Image.Image) -> float:
    filtered = image.convert("L").filter(LAPLACIAN_KERNEL)
    return float(np.asarray(filtered, dtype=np.float64).var())
