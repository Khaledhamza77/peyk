from .base import Cell, TableStructure, TSRBackend
from .tatr import TATRBackend
from .rapidtable import RapidTableBackend
from .pp_structure import PPStructureGeneralBackend, PPStructureWiringBackend
from .tableformer import TableFormerBackend

BACKENDS = {
    "tatr": TATRBackend,
    "rapidtable": RapidTableBackend,
    "pp-structure-general": PPStructureGeneralBackend,
    "pp-structure-wiring": PPStructureWiringBackend,
    "tableformer": TableFormerBackend,
}

__all__ = ["Cell", "TableStructure", "TSRBackend", "BACKENDS"]
