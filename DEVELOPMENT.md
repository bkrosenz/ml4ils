# Development Guide for ml4ils

This guide covers the development workflow, testing practices, and architecture for contributors.

## Quick Start for Contributors

```bash
# Clone the repository
git clone https://github.com/bkrosenz/ml4ils.git
cd ml4ils

# Create a development environment
pip install -e ".[scientific,test]"

# Install development tools
pip install black isort mypy pytest-cov

# Verify setup
pytest tests/
mypy ml4ils/
```

## Project Architecture

### Package Structure

```
ml4ils/                          # Production package (v0.1.0)
├── __init__.py                 # Package metadata
├── models.py                   # Core PyTorch models and metrics
├── meta_to_hdf.py              # HDF5 dataset builder
├── sql_utils.py                # Database utilities (lazy wrapper)
├── stat_utils.py               # Statistics (lazy wrapper)
├── utils.py                    # Common utilities (lazy wrapper)
├── _config.py                  # Configuration helpers
├── _legacy.py                  # Lazy loading mechanism
├── plotting/                   # Visualization modules
│   ├── __init__.py
│   └── plot_*.py               # Various plotting utilities
└── [other analysis modules]    # Individual analysis entry points

code/                            # Legacy code (backward compatible)
├── pylib/                       # Original Python utilities
├── shell_scripts/               # Pipeline orchestration
├── plotting/                    # Legacy plotting scripts
└── simulate/                    # Simulation code

tests/                           # Automated tests
├── test_package.py             # Package validation
└── [future test modules]

docs/                            # (Planned) Sphinx documentation
```

### Design Principles

1. **Package First**: Reusable logic lives in `ml4ils/` with stable APIs
2. **Backward Compatible**: Legacy `code/pylib/` imports remain functional via lazy wrappers
3. **Side-Effect Free**: Imports should not execute models or queries
4. **Explicit APIs**: Type hints and docstrings for all public functions
5. **Lazy Loading**: Heavy dependencies (database, simulation tools) loaded on-demand
6. **Testable**: All functionality has corresponding tests with clear assertions

## Development Workflow

### 1. Set Up Development Environment

```bash
# Activate the package in editable mode
pip install -e ".[scientific,test]"

# Install optional development tools
pip install black isort mypy pytest-cov
```

### 2. Make Changes

#### Adding a New Function to `models.py`

```python
# ml4ils/models.py
def my_new_metric(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute my new metric.
    
    Args:
        logits: Model predictions. Shape: (N,)
        target: True labels. Shape: (N,)
    
    Returns:
        Scalar metric value.
    """
    # Implementation
    pass
```

#### Creating a New Module

```bash
# Option 1: New reusable module (add to ml4ils/)
touch ml4ils/my_module.py

# Option 2: New lazy wrapper for legacy code
# Edit ml4ils/my_module.py:
"""Module docstring."""
from ._legacy import load_legacy

def __getattr__(name):
    return getattr(load_legacy("my_module"), name)

def __dir__():
    return sorted(set(globals()) | set(dir(load_legacy("my_module"))))
```

#### Modifying Existing Logic

**Before**:
- Carefully review where the function is called
- Check for tests that validate current behavior
- Look for database or file dependencies

**During**:
- Keep changes minimal and focused
- Add type hints compatible with Python 3.9+
- Preserve backward compatibility

**After**:
- Add or update tests
- Run the full test suite
- Check that legacy imports still work

### 3. Testing

#### Run All Tests

```bash
pytest tests/ -v
```

#### Run Specific Test

```bash
# Test models module only
pytest tests/test_package.py::test_classification_metrics_are_stable_for_binary_labels -v
```

#### Test with Coverage

```bash
pytest tests/ --cov=ml4ils --cov-report=html
# Open htmlcov/index.html
```

#### Add New Tests

```python
# tests/test_my_feature.py
import pytest
from ml4ils.models import my_new_metric

def test_my_new_metric_computes_correctly():
    import torch
    logits = torch.tensor([1.0, -1.0])
    target = torch.tensor([1, 0])
    result = my_new_metric(logits, target)
    assert torch.isclose(result, torch.tensor(expected_value))

def test_my_new_metric_handles_edge_cases():
    import torch
    # All same class
    logits = torch.tensor([1.0, 1.0])
    target = torch.tensor([1, 1])
    result = my_new_metric(logits, target)
    # Should not raise or return nan
    assert torch.isfinite(result)
```

### 4. Code Quality

#### Format Code

```bash
# Format all Python files
black ml4ils/

# Sort imports
isort ml4ils/
```

#### Check Types

```bash
# Type check the package
mypy ml4ils/

# With verbose output
mypy ml4ils/ --show-error-codes
```

#### Lint

```bash
# Check style (optional)
pip install flake8
flake8 ml4ils/ --max-line-length=100
```

### 5. Commit and Push

```bash
# Verify changes
git status
git diff

# Stage changes
git add ml4ils/ tests/

# Commit with descriptive message
git commit -m "Add my_new_metric function to models module

- Implements symmetric metric for class imbalance
- Handles binary and multi-class predictions
- Includes gradient clipping support
- Tests: test_my_new_metric_computes_correctly"

# Push to your fork or branch
git push origin feature-my-new-metric
```

## Release Process

### Version Numbering

Uses semantic versioning: `MAJOR.MINOR.PATCH`

- `MAJOR`: Incompatible API changes
- `MINOR`: New features, backward compatible
- `PATCH`: Bug fixes

Current version: See `ml4ils/__init__.py` and `pyproject.toml`

### Steps to Release

1. **Update version** in `ml4ils/__init__.py` and `pyproject.toml`
2. **Update CHANGELOG** with new features and fixes
3. **Run full test suite**: `pytest tests/`
4. **Check type safety**: `mypy ml4ils/`
5. **Create GitHub release** with tag `v0.2.0`
6. **Build distribution**: `python -m build`
7. **Upload to PyPI**: `twine upload dist/*`

## Database Setup (Optional)

To work with PostgreSQL features:

```bash
# Install database adapter
pip install psycopg2-binary

# Create database (one-time)
createdb ml4ils

# Connect via Python
from ml4ils.sql_utils import connect_to_db
conn = connect_to_db()
```

See `code/shell_scripts/create_views.sql` for schema.

## Profiling and Benchmarking

### Profile Runtime

```python
import cProfile
import pstats
from ml4ils.models import Net

profiler = cProfile.Profile()
profiler.enable()

# Your code here
net = Net(50, [100, 50])
# ... training loop ...

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative').print_stats(20)
```

### Profile Memory

```python
from memory_profiler import profile

@profile
def my_expensive_function():
    # Code to profile
    pass

# Run: python -m memory_profiler script.py
```

### Benchmark Comparisons

```bash
# Before optimization
time python code/train_config.py --config configs/baseline.config

# After optimization
time python code/train_config.py --config configs/baseline.config
```

## Troubleshooting

### Issue: Import Errors

```python
# Check what's importable
import ml4ils
dir(ml4ils)

# Check lazy wrappers
from ml4ils.stat_utils import __dir__
print(__dir__())
```

### Issue: Tests Fail

```bash
# Run with verbose output
pytest tests/ -vv -s

# Run single test to debug
pytest tests/test_package.py::specific_test -vv --pdb

# Check test isolation
pytest tests/ --forked  # requires pytest-forked
```

### Issue: Type Checking Fails

```bash
# Show detailed type errors
mypy ml4ils/ --show-error-codes --show-traceback

# Ignore specific line
x = some_func()  # type: ignore
```

### Issue: Database Queries Fail

```python
# Check connection
from ml4ils.sql_utils import connect_to_db
with connect_to_db() as conn:
    cur = conn.cursor()
    cur.execute("SELECT version();")
    print(cur.fetchone())
```

## Contributing Guidelines

### Before Submitting a PR

- [ ] Code follows black/isort style (run `black ml4ils/ && isort ml4ils/`)
- [ ] Type hints added to new functions (`mypy ml4ils/`)
- [ ] Docstrings written for public APIs
- [ ] Tests added/updated for new behavior
- [ ] All tests pass: `pytest tests/`
- [ ] Backward compatibility verified (legacy imports still work)
- [ ] No breaking changes to CLI arguments or data formats
- [ ] Commit message is descriptive and references related issues

### Commit Message Template

```
[component] Brief description

Detailed explanation of what changed and why.

- List of specific changes
- Impact on APIs or behavior
- Any relevant performance notes

Fixes #123
Related to #456
```

### Code Review Expectations

- Clear, testable changes
- Docstrings and type hints for public APIs
- Tests for edge cases and error conditions
- No regressions in existing functionality
- Performance impact discussed if significant
- Backward compatibility preserved

## Documentation

### Update API Docs

If you add public functions to `ml4ils/models.py` or other core modules:

1. Add docstring with Args, Returns, Examples
2. Update [API.md](API.md) with new function signature
3. Add test to [tests/test_package.py](tests/test_package.py)

### Update README

If you change project structure, dependencies, or major features:

1. Update [README.md](README.md) overview
2. Update [INSTALL.md](INSTALL.md) installation steps if needed
3. Add example usage to [example/](example/) if applicable

## Future Roadmap

- [ ] Add more comprehensive test coverage
- [ ] Add CI/CD pipeline (GitHub Actions)
- [ ] Generate Sphinx documentation site
- [ ] Add performance benchmarks
- [ ] Implement caching layer for database queries
- [ ] Add optional Dask support for large-scale processing
- [ ] Port simulation code (`code/simulate/`) to package

## Resources

- **Issue Tracker**: GitHub Issues
- **Discussions**: GitHub Discussions
- **Documentation**: [README.md](README.md), [INSTALL.md](INSTALL.md), [API.md](API.md)
- **Wiki**: https://github.com/bkrosenz/ml4ils/wiki
- **Examples**: [example/](example/), [code/shell_scripts/](code/shell_scripts/)
- **Tests**: [tests/](tests/)

## Quick Reference

| Task | Command |
|------|---------|
| Install dev env | `pip install -e ".[scientific,test]"` |
| Run tests | `pytest tests/ -v` |
| Format code | `black ml4ils/` |
| Check types | `mypy ml4ils/` |
| Test coverage | `pytest --cov=ml4ils` |
| Build wheel | `python -m build` |
| Run example | `bash example/predict.sh` |

---

**Questions?** Open an issue or discussion on GitHub!
