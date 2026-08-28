"""Fractional SCF summary compatibility API."""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("get_summary_stats_frac_scf"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("get_summary_stats_frac_scf"))))
