"""Repository configuration helpers for the installable ml4ils package."""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """Return the repository root, honoring ML4ILS_ROOT when provided."""
    env_root = os.environ.get("ML4ILS_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    return Path(__file__).resolve().parent.parent


def legacy_root() -> Path:
    """Return the legacy ``code/pylib`` directory used by compatibility imports."""
    return repo_root() / "code" / "pylib"
