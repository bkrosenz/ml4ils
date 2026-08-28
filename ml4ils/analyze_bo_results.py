"""Bayesian-optimization result analysis compatibility API."""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("analyze_bo_results"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("analyze_bo_results"))))
