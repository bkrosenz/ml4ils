"""Lazy loading support for the repository's legacy script modules."""

from importlib import import_module
from pathlib import Path
import sys
from types import ModuleType
from typing import Dict

_LEGACY_ROOT = Path(__file__).resolve().parent.parent / "code" / "pylib"
_CACHE: Dict[str, ModuleType] = {}


def load_legacy(name: str) -> ModuleType:
    """Load a legacy ``code/pylib`` module without importing it at package startup."""
    if name not in _CACHE:
        legacy_path = str(_LEGACY_ROOT)
        if legacy_path not in sys.path:
            sys.path.insert(0, legacy_path)
        _CACHE[name] = import_module(name)
    return _CACHE[name]
