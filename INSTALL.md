# Installation Guide for ml4ils

This guide covers installation for both package users and developers.

## Prerequisites

- **Python 3.9 or later** (Python 3.12 recommended)
- **pip** or **conda** for package management
- **git** for cloning the repository

## Package Installation (Users)

### Option 1: Development Install (Recommended)

Clone the repository and install in editable mode with scientific dependencies:

```bash
git clone https://github.com/bkrosenz/ml4ils.git
cd ml4ils
pip install -e ".[scientific,test]"
```

This installs:
- **ml4ils** package (editable, pulls from source)
- **scientific** extras: numpy, pandas, scikit-learn, scipy, torch
- **test** extras: pytest (for running tests)

### Option 2: Installation from PyPI (When Published)

```bash
pip install ml4ils[scientific,test]
```

### Option 3: Legacy Environment with Conda

For compatibility with existing shell scripts and code under `code/`:

```bash
conda env create -f code/conda-env.yml
conda activate ml4ils
```

**Note**: This environment uses Python 3.8 and legacy dependencies. Use **Option 1** for new work.

## Verifying Installation

### Test Package Import

```python
python -c "import ml4ils; print(ml4ils.__version__)"
# Output: 0.1.0
```

### Run Test Suite

```bash
pytest tests/ -v
```

Expected output:
```
tests/test_package.py::test_package_metadata PASSED
tests/test_package.py::test_classification_metrics_are_stable_for_binary_labels PASSED
tests/test_package.py::test_legacy_adapters_are_lazy PASSED
tests/test_package.py::test_model_api_and_legacy_wrapper PASSED
```

### Quick API Check

```python
import torch
from ml4ils.models import Net, calculate_f1

# Create a simple model
model = Net(input_size=50, hidden_layers=[100, 50])
logits = model(torch.randn(10, 50))
target = torch.randint(0, 2, (10,))

# Compute F1 score
f1 = calculate_f1(logits, target, pos=1)
print(f"F1 Score: {f1:.4f}")
```

## Development Installation

For contributors wanting to modify the package:

### 1. Clone and Install in Editable Mode

```bash
git clone https://github.com/bkrosenz/ml4ils.git
cd ml4ils
pip install -e ".[scientific,test]"
```

### 2. Install Development Tools (Optional)

For code formatting, type checking, and profiling:

```bash
pip install black isort mypy pytest-cov
```

### 3. Verify the Setup

```bash
# Run tests with coverage
pytest tests/ --cov=ml4ils

# Check type hints
mypy ml4ils/

# Format code
black ml4ils/
isort ml4ils/
```

See [DEVELOPMENT.md](DEVELOPMENT.md) for the full developer workflow.

## Advanced: Manual Dependency Installation

If you prefer manual installation or have specific version requirements:

```bash
# Core dependencies
pip install numpy>=1.20 pandas>=1.4 scikit-learn>=1.0 scipy>=1.7

# Deep learning (required for models.py)
pip install torch>=1.10

# Testing (optional)
pip install pytest>=7

# Development tools (optional)
pip install mypy black isort
```

## Troubleshooting

### Issue: `ModuleNotFoundError: No module named 'ml4ils'`

**Solution**: Ensure the package is installed in editable mode:
```bash
pip install -e ".[scientific,test]"
```

Verify with:
```bash
pip show ml4ils
```

### Issue: PyTorch Installation Fails

**Solution**: Install PyTorch matching your hardware. For CPU-only:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

For GPU (CUDA 11.8):
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

See https://pytorch.org/get-started/locally/ for your system configuration.

### Issue: Tests Fail with `torch is optional`

**Solution**: Some tests are skipped if torch is not installed. Reinstall with scientific dependencies:
```bash
pip install -e ".[scientific]"
```

### Issue: PostgreSQL Database Tests Fail

**Solution**: Database tests are optional. Install optional database dependencies if needed:
```bash
pip install psycopg2-binary
```

## Using Pretrained Models

Pretrained models are stored in the `models/` directory:

```python
import torch
from pathlib import Path

# Load a pretrained model
model_config_path = Path("models/DNN-Pred/model.config")
model_path = Path("models/DNN-Pred/model_state_dict.pt")

# Parse config and load model
# (See predict.sh example for full workflow)
```

## Using Legacy Scripts

Existing shell scripts and Python code under `code/` continue to work:

```bash
# Example: Run legacy inference pipeline
bash example/infer_trees.sh

# Example: Run legacy prediction
bash example/predict.sh
```

**Note**: Legacy scripts may contain hardcoded paths. Update paths as needed for your system.

## Environment Variables

Optional environment variables for configuration:

```bash
# Set random seed for reproducibility
export ML4ILS_SEED=42

# PostgreSQL connection (if using database features)
export PGHOST=localhost
export PGPORT=5432
export PGUSER=your_username
export PGPASSWORD=your_password
export PGDATABASE=ml4ils
```

## Next Steps

- **Users**: See [README.md](README.md) for quick-start examples and [API.md](API.md) for module reference
- **Developers**: See [DEVELOPMENT.md](DEVELOPMENT.md) for contribution guidelines
- **Detailed Examples**: Check [example/](example/) for end-to-end workflows
- **Wiki**: Visit the [project wiki](https://github.com/bkrosenz/ml4ils/wiki) for comprehensive tutorials
