"""Shared phylogenetic and dataset utilities.

The implementation remains in ``code/pylib/utils.py`` during the compatibility
migration; this module provides the installable package import path.
"""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("utils"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("utils"))))
