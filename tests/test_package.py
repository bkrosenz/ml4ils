import importlib.util

import pytest


def test_package_metadata():
    import ml4ils

    assert ml4ils.__version__ == "0.1.0"

    
def test_legacy_adapters_are_lazy():
    import importlib
    import sys

    module_names = (
        "consolidate_inferred_trees",
        "consolidate_scf",
        "consolidate_trees",
        "get_recomb_stats",
        "get_summary_stats_frac_scf",
        "get_summary_stats_heterotachy",
        "get_summary_stats_model_scf",
        "gmean",
        "learn_simphy",
        "meta_to_hdf",
        "sql_utils",
        "stat_utils",
        "summarize_meta",
        "train_and_test_new",
        "utils",
    )
    for name in module_names:
        module = importlib.import_module("ml4ils." + name)
        assert name not in sys.modules
        assert callable(module.__dir__)


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None, reason="torch is optional"
)
def test_model_api_and_legacy_wrapper():
    import sys

    import torch

    from ml4ils.models import Net

    sys.path.insert(0, "code/pylib")
    try:
        import models as legacy_models

        assert legacy_models.Net is Net
        assert tuple(Net(2, [3, 1])(torch.ones(4, 2)).shape) == (4, 1)
    finally:
        sys.path.pop(0)
