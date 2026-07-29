import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from PIL import Image

from backends.table_scale import MAX_PIXELS, MIN_SCALE, PIXEL_SAFETY_MARGIN, safe_upscale


def test_small_image_gets_upscaled_toward_the_cap():
    image = Image.new("RGB", (500, 300), "white")  # 150,000px, tiny relative to MAX_PIXELS
    result = safe_upscale(image, max_upscale_cap=2.0)
    assert result.size == (1000, 600)  # exactly 2.0x, the cap — plenty of pixel headroom below it


def test_upscale_never_exceeds_the_cap_even_with_huge_headroom():
    image = Image.new("RGB", (10, 10), "white")
    result = safe_upscale(image, max_upscale_cap=1.5)
    assert result.size == (15, 15)


def test_image_already_near_the_pixel_ceiling_is_left_unchanged():
    # sqrt(MAX_PIXELS * PIXEL_SAFETY_MARGIN) ~= 2470px per side for a square image -> scale_cap ~= 1.0
    side = int((MAX_PIXELS * PIXEL_SAFETY_MARGIN) ** 0.5) + 1
    image = Image.new("RGB", (side, side), "white")
    result = safe_upscale(image)
    assert result is image  # MIN_SCALE floor: never downscale, and no meaningful upscale headroom either


def test_never_downscales_a_crop_already_over_the_pixel_ceiling():
    # Deliberately oversized crop (scale_cap < 1.0) — must still never shrink it.
    side = int((MAX_PIXELS * PIXEL_SAFETY_MARGIN) ** 0.5) + 500
    image = Image.new("RGB", (side, side), "white")
    result = safe_upscale(image)
    assert result is image
    assert result.size == (side, side)
