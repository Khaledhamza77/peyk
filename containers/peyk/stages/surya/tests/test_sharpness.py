import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from PIL import Image, ImageFilter

from backends.sharpness import laplacian_variance


def _checkerboard(size=1000, square=10):
    """A sharp, high-frequency image: alternating black/white squares, built with numpy for
    speed at a size large enough that PIL's border-pixel handling (which varies across
    versions for a negative-coefficient kernel) doesn't dominate the result. Should score
    high."""
    idx = np.arange(size) // square
    grid = (idx[:, None] + idx[None, :]) % 2
    arr = np.where(grid == 0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def test_sharp_image_scores_high():
    sharp = _checkerboard()
    assert laplacian_variance(sharp) > 5000


def test_blurring_reduces_the_score():
    sharp = _checkerboard()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=3))
    assert laplacian_variance(blurred) < laplacian_variance(sharp) / 10


def test_flat_image_scores_much_lower_than_a_sharp_image():
    # Not an absolute near-zero assertion: PIL's border-pixel handling for this
    # negative-coefficient kernel contributes a small nonzero variance for ANY image (varies
    # by Pillow version) that a tiny synthetic image can't distinguish from real signal. A
    # wide relative margin against the sharp image is robust to that and still verifies the
    # property that actually matters: a flat image scores nowhere near as sharp.
    sharp = _checkerboard()
    flat = Image.new("RGB", (1000, 1000), (255, 255, 255))
    assert laplacian_variance(flat) < laplacian_variance(sharp) / 10
