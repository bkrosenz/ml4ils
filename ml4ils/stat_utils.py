"""Statistical helpers exposed through the installable package."""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("stat_utils"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("stat_utils"))))
