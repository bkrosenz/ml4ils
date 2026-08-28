"""Topology-frequency plotting compatibility API."""

from .._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("plot_topo_freqs2"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("plot_topo_freqs2"))))
