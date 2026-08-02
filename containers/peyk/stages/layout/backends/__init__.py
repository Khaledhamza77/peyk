from .base import LayoutBackend, Region
from .pp_doclayout import PPDocLayoutV2Backend
from .surya_layout import SuryaLayoutBackend
from .doclayout_yolo import DocLayoutYOLOBackend
from .heron import HeronBackend

BACKENDS = {
    "pp-doclayout-v2": PPDocLayoutV2Backend,
    "surya-layout": SuryaLayoutBackend,
    "doclayout-yolo": DocLayoutYOLOBackend,
    "heron": HeronBackend,
}

__all__ = ["LayoutBackend", "Region", "BACKENDS"]
