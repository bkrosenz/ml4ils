"""Compatibility wrapper for the packaged PyTorch model APIs."""

from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ml4ils.models import *  # noqa: F401,F403,E402
