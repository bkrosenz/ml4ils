"""Lazy loading support for the repository's legacy script modules."""

from __future__ import annotations

from importlib import import_module
import sys
from types import ModuleType
from typing import Dict

from ._config import legacy_root

_CACHE: Dict[str, ModuleType] = {}


def load_legacy(name: str) -> ModuleType:
    """Load a legacy ``code/pylib`` module without importing it at package startup."""
    if name not in _CACHE:
        legacy_path = str(legacy_root())
        if legacy_path not in sys.path:
            sys.path.insert(0, legacy_path)
        _CACHE[name] = import_module(name)
    return _CACHE[name]
