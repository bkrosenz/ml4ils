"""Condition-level plotting compatibility API."""

from .._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("plot_ml_by_condition"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("plot_ml_by_condition"))))
