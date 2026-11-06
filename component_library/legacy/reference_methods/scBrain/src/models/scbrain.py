"""Public scBrain model aliases.

The original experiment code used `mjm_*` module names.  The paper and release
use `scBrain` as the public method name, while retaining the legacy classes for
backward-compatible log parsing and checkpoint reproduction.
"""

from src.models.mjm import FFN, RMSNorm, SwiGLU
from src.models.mjm import mjm as scBrainFlat
from src.models.mjm import mjm_1 as scBrain
from src.models.mjm_2 import mjm_2a as scBrainDepth
from src.models.mjm_2 import mjm_2b as scBrainMorphoSpatial
from src.models.mjm_2 import mjm_2c as scBrainFiLM
from src.models.mjm_nlevel import mjm_1_nlevel as scBrainNLevel

__all__ = [
    "RMSNorm",
    "SwiGLU",
    "FFN",
    "scBrain",
    "scBrainFlat",
    "scBrainDepth",
    "scBrainMorphoSpatial",
    "scBrainFiLM",
    "scBrainNLevel",
]
