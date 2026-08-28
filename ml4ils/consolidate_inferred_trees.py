"""Inferred-tree consolidation compatibility API."""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("consolidate_inferred_trees"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("consolidate_inferred_trees"))))
