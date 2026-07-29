"""Q3's upscale rule (docs-personal/surya/improvement.md): the largest safe upscale under
peyk-vllm-surya's max_pixels cap, never downscaling. max_upscale_cap=2.0 is a
confirmed-necessary defensive ceiling, not arbitrary caution — removing it reproduces a hard
"Inference error: 'NoneType' object has no attribute 'chat'" failure when a chunk's actual
pixel count (after independently rounding width/height up) creeps past the real cap. Ported
from z_surya_exploration/process_table.py's safe_upscale."""
import math

from PIL import Image

MAX_PIXELS = 6_291_456  # peyk-vllm-surya's --mm-processor-kwargs max_pixels (start.sh) — do not change without re-verifying against that config
PIXEL_SAFETY_MARGIN = 0.97
MAX_UPSCALE_CAP = 2.0
MIN_SCALE = 1.0


def safe_upscale(image: Image.Image, max_upscale_cap: float = MAX_UPSCALE_CAP) -> Image.Image:
    w, h = image.size
    scale_cap = math.sqrt(MAX_PIXELS * PIXEL_SAFETY_MARGIN / (w * h))
    scale = min(scale_cap, max_upscale_cap)
    if scale <= MIN_SCALE:
        return image

    new_size = (round(w * scale), round(h * scale))
    return image.resize(new_size, Image.LANCZOS)
