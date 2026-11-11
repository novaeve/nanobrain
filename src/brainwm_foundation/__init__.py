"""Standalone import surface for the new unified BrainWM foundation."""
from pathlib import Path
import importlib.util
import sys

_root = Path(__file__).resolve().parents[1] / 'brainwm'
__path__.append(str(_root))
_spec = importlib.util.spec_from_file_location('brainwm_foundation.foundation_moe', _root / 'foundation_moe.py')
_module = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = _module
_spec.loader.exec_module(_module)
BrainWMInput = _module.BrainWMInput
BrainWMFoundationMoE = _module.BrainWMFoundationMoE
parameter_count = _module.parameter_count
freeze_for_modality_adaptation = _module.freeze_for_modality_adaptation
from .backends import (
    backend_catalog, backend_names, backend_spec, create_backend,
    load_backend_entry_points, load_backend_plugins,
)

__all__ = [
    'BrainWMInput', 'BrainWMFoundationMoE', 'parameter_count',
    'freeze_for_modality_adaptation', 'backend_catalog', 'backend_names',
    'backend_spec', 'create_backend',
    'load_backend_entry_points', 'load_backend_plugins',
]
