"""Metazoa test workflow compatibility API."""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("test_metazoa"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("test_metazoa"))))
