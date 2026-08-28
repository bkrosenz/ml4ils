"""Metadata conversion workflow compatibility API."""

from ._legacy import load_legacy


def __getattr__(name):
    return getattr(load_legacy("meta_to_hdf"), name)


def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("meta_to_hdf"))))
