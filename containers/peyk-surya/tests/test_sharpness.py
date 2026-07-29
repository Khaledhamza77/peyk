import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image, ImageFilter

from backends.sharpness import laplacian_variance


def _checkerboard(size=200, square=10):
    """A sharp, high-frequency image: alternating black/white squares. Should score high."""
    image = Image.new("L", (size, size), 255)
    pixels = image.load()
    for y in range(size):
        for x in range(size):
            if (x // square + y // square) % 2 == 0:
                pixels[x, y] = 0
    return image.convert("RGB")


def test_sharp_image_scores_high():
    sharp = _checkerboard()
    assert laplacian_variance(sharp) > 1000


def test_blurring_reduces_the_score():
    sharp = _checkerboard()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=3))
    assert laplacian_variance(blurred) < laplacian_variance(sharp) / 10


def test_flat_image_scores_near_zero():
    flat = Image.new("RGB", (200, 200), (255, 255, 255))
    assert laplacian_variance(flat) < 1.0
