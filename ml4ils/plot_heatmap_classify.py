"""Classification heatmap workflow compatibility API."""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("plot_heatmap_classify"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("plot_heatmap_classify"))))
