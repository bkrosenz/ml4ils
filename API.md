# API Reference for ml4ils

Complete reference for publicly available modules and functions in the ml4ils package.

## Core Package

### Module: `ml4ils`

**Version**: `ml4ils.__version__` → "0.1.0"

## Models (`ml4ils.models`)

Core PyTorch models and evaluation metrics for phylogenetic inference tasks.

### Classes

#### `Net`

```python
class Net(nn.Sequential):
    """Feed-forward neural network for phylogenomic sequence features.
    
    A flexible PyTorch Sequential model that automatically constructs layers,
    applies activations, and returns log-probabilities for classification or
    probabilistic output for regression.
    """
    
    def __init__(
        self,
        input_size: int,
        layer_sizes: Optional[Iterable[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        """Construct the network.
        
        Args:
            input_size: Width of input feature vectors.
            layer_sizes: Widths of hidden and output layers.
                Defaults to [120, 849, 1].
                Final layer size determines output:
                  - 1: Sigmoid activation (binary/regression)
                  - >1: LogSoftmax activation (multi-class)
            dropout: Dropout probability (currently unused).
        
        Raises:
            ValueError: If layer_sizes is empty.
        
        Example:
            >>> model = Net(input_size=50, layer_sizes=[100, 50, 2])
            >>> logits = model(torch.randn(10, 50))
            >>> logits.shape
            torch.Size([10, 2])
        """
```

### Functions

#### `calculate_f1(logits, target, pos=0, thresh=0.5) -> Tensor`

```python
def calculate_f1(
    log_proba: torch.Tensor,
    target: torch.Tensor,
    pos: int = 0,
    thresh: float = 0.5,
) -> torch.Tensor:
    """Compute F1-score for a positive class.
    
    Handles both binary predictions (threshold-based) and multi-class
    (argmax-based).
    
    Args:
        log_proba: Predicted logits or log-probabilities. Shape: (N,) for binary,
            (N, C) for multi-class.
        target: True class labels. Shape: (N,).
        pos: Class index to treat as positive. Default: 0.
        thresh: Decision threshold for binary classification. Default: 0.5.
    
    Returns:
        F1-score as a scalar tensor on the same device as target.
    
    Example:
        >>> logits = torch.tensor([2.0, -1.0, 0.5, -0.5])
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> f1 = calculate_f1(logits, target, pos=1)
    """
```

#### `calculate_mcc(logits, target, pos=0, thresh=0.5) -> Tensor`

```python
def calculate_mcc(
    log_proba: torch.Tensor,
    target: torch.Tensor,
    pos: int = 0,
    thresh: float = 0.5,
) -> torch.Tensor:
    """Compute Matthews Correlation Coefficient.
    
    Provides a balanced measure of classifier quality even with class imbalance.
    
    Args:
        log_proba: Predicted logits or log-probabilities. Shape: (N,) for binary,
            (N, C) for multi-class.
        target: True class labels. Shape: (N,).
        pos: Class index to treat as positive. Default: 0.
        thresh: Decision threshold for binary classification. Default: 0.5.
    
    Returns:
        MCC as a scalar tensor in [-1, 1] on the same device as target.
        Returns 0 if denominator is zero (e.g., all predictions same as all targets).
    
    Example:
        >>> logits = torch.tensor([2.0, -1.0, 0.5, -0.5])
        >>> target = torch.tensor([1, 1, 0, 0])
        >>> mcc = calculate_mcc(logits, target, pos=1)
    """
```

#### `calculate_accuracy(logits, target) -> Tensor`

```python
def calculate_accuracy(
    log_proba: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute classification accuracy.
    
    Args:
        log_proba: Predicted logits. Shape: (N,) for binary, (N, C) for multi-class.
        target: True class labels. Shape: (N,).
    
    Returns:
        Accuracy as a scalar tensor in [0, 1].
    """
```

#### `instantiate_model(dim, layers, clip_value=10) -> Net`

```python
def instantiate_model(
    dim: int,
    layers: list[int],
    clip_value: float = 10,
) -> Net:
    """Build a model with gradient clipping.
    
    Constructs a network and attaches hooks to clip gradients during backprop.
    Uses DataParallel if multiple GPUs available.
    
    Args:
        dim: Input feature dimension.
        layers: Hidden and output layer widths.
        clip_value: Max gradient magnitude to allow. Default: 10.
    
    Returns:
        A Net instance placed on the appropriate device.
    """
```

#### `load_model(config, dataset, checkpoint_dir=None, with_optimizer=False, optimizer=optim.Adam) -> Net | Tuple[Net, Optimizer]`

```python
def load_model(
    config: dict,
    dataset,
    checkpoint_dir: Path | None = None,
    with_optimizer: bool = False,
    optimizer=optim.Adam,
):
    """Load a network and optionally restore state from checkpoint.
    
    Args:
        config: Dictionary with keys 'layers' (list of ints) and 'lr' (float).
        dataset: Either an integer (input dimension) or a dataset object
            where dataset[0][0].shape[0] gives input dimension.
        checkpoint_dir: Path to directory containing 'checkpoint' file.
        with_optimizer: If True, also restore optimizer state.
        optimizer: Optimizer class (e.g., optim.Adam). Default: Adam.
    
    Returns:
        Net instance if with_optimizer=False.
        Tuple of (Net, Optimizer) if with_optimizer=True.
    
    Example:
        >>> config = {'layers': [100, 50, 1], 'lr': 0.001}
        >>> net = load_model(config, dataset=50)
        >>> net, opt = load_model(config, dataset=50, with_optimizer=True,
        ...                        checkpoint_dir=Path('checkpoints'))
    """
```

#### `get_device() -> str`

```python
def get_device() -> str:
    """Determine the preferred PyTorch device.
    
    Returns:
        'cuda:0' if GPU is available, otherwise 'cpu'.
    """
```

## Data Processing (`ml4ils.meta_to_hdf`)

Utilities for converting raw phylogenetic data to HDF5 format.

```python
from ml4ils.meta_to_hdf import MetaHDFBuilder
```

### Class: `MetaHDFBuilder`

Constructs HDF5 datasets from phylogenetic metadata and summary statistics.
Supports batching, parallel processing, and resumable writes.

See [example/infer_trees.sh](example/infer_trees.sh) for usage examples.

## Database Utilities (`ml4ils.sql_utils`)

Lazy wrappers for querying phylogenetic simulation databases.

```python
from ml4ils.sql_utils import query_simulations, fetch_trees, ...
```

All functions from `code/pylib/sql_utils.py` are available via lazy import.

## Statistical Utilities (`ml4ils.stat_utils`)

Statistical functions for phylogenetic and tree metrics.

```python
from ml4ils.stat_utils import compute_scf, compute_rf_distance, ...
```

All functions from `code/pylib/stat_utils.py` are available via lazy import.

## General Utilities (`ml4ils.utils`)

Shared phylogenetic and dataset utilities.

```python
from ml4ils.utils import read_newick, write_newick, ...
```

All functions from `code/pylib/utils.py` are available via lazy import.

## Analysis Modules

The following modules provide reusable entry points for common analysis tasks.
Each can be imported as a module and called with appropriate configuration.

### Available Modules

- `ml4ils.get_recomb_stats` — Recombination statistics computation
- `ml4ils.get_summary_stats_frac_scf` — Fraction of sites with conflicting signal
- `ml4ils.get_summary_stats_heterotachy` — Heterotachy metrics
- `ml4ils.get_summary_stats_model_scf` — Model-based SCF statistics
- `ml4ils.consolidate_trees` — Merge tree inference results
- `ml4ils.consolidate_scf` — Aggregate site concordance factors
- `ml4ils.consolidate_inferred_trees` — Organize inferred tree outputs
- `ml4ils.summarize_meta` — Summarize simulation metadata
- `ml4ils.learn_simphy` — SimPhy simulation interface
- `ml4ils.gmean` — Geometric mean calculations
- `ml4ils.train_and_test_new` — Training pipeline orchestration
- `ml4ils.predict_config` — Prediction with model configuration
- `ml4ils.test_config` — Test/evaluation configuration
- `ml4ils.test_metazoa` — Metazoa dataset tests
- `ml4ils.train_config` — Training configuration management
- `ml4ils.analyze_bo_results` — Bayesian optimization result analysis
- `ml4ils.bo_learn_torch` — Bayesian optimization with PyTorch
- `ml4ils.bo_learn_torch_topo` — Topology-focused Bayesian optimization
- `ml4ils.plot_heatmap_classify` — Classification heatmap plotting
- `ml4ils.plotting.plot_heatmap` — Generic heatmap utilities
- `ml4ils.plotting.plot_metazoa` — Metazoa-specific plots
- `ml4ils.plotting.plot_ml_by_condition` — Condition-stratified ML plots
- `ml4ils.plotting.plot_test_results` — Test result visualization
- `ml4ils.plotting.plot_topo_freqs2` — Topology frequency plots

## Backward Compatibility

Legacy imports from `code/pylib/` continue to work for compatibility:

```python
# All equivalent to the ml4ils.* equivalents above
import sys
sys.path.insert(0, 'code/pylib')

from models import Net, calculate_f1, calculate_mcc
from sql_utils import query_simulations
from utils import read_newick
# ... etc
```

## Configuration (`ml4ils._config`)

Package-level configuration defaults and helpers.

```python
from ml4ils._config import get_config, set_config
```

## Type Hints

All public APIs use type hints compatible with Python 3.9+:

```python
from typing import Optional, Iterable, Tuple
from pathlib import Path
import torch
import torch.nn as nn
```

## Error Handling

Functions raise standard Python exceptions:

- `ValueError`: Invalid argument values (e.g., empty layer_sizes)
- `FileNotFoundError`: Missing checkpoint or data files
- `RuntimeError`: PyTorch device errors or invalid tensor operations
- `IOError`: Database or file access errors

## Performance Notes

- **GPU Acceleration**: `Net` and metrics automatically use GPU if available
- **Memory**: Large HDF5 datasets are processed in chunks; see `MetaHDFBuilder`
- **Parallelization**: Database queries and file I/O support multi-threading
- **Reproducibility**: Use `get_device()` and pin seeds for determinism

## Additional Resources

- **Examples**: See [example/](example/) for end-to-end workflows
- **Tests**: See [tests/test_package.py](tests/test_package.py) for API usage examples
- **Legacy Scripts**: See [code/shell_scripts/](code/shell_scripts/) for pipeline examples
- **Wiki**: Full tutorials at https://github.com/bkrosenz/ml4ils/wiki
