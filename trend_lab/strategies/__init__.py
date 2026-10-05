"""Registry of every strategy family. New batches get appended here."""

from .families_v1 import FAMILIES as V1
from .families_v2 import FAMILIES as V2
from .families_v3 import FAMILIES as V3

ALL_FAMILIES = {f.name: f for f in V1 + V2 + V3}
