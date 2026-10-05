"""Registry of every strategy family. New batches get appended here."""

from .families_v1 import FAMILIES as V1
from .families_v2 import FAMILIES as V2
from .families_v3 import FAMILIES as V3
from .families_v4 import FAMILIES as V4
from .families_v5 import FAMILIES as V5
from .families_v6 import FAMILIES as V6
from .families_v7 import FAMILIES as V7

ALL_FAMILIES = {f.name: f for f in V1 + V2 + V3 + V4 + V5 + V6 + V7}
