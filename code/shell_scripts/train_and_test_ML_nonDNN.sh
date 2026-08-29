# syntax: bash train_and_test_ML_nonDNN.sh <path_to_dir_containing_config> <num_procs>
# environment override: export ML4ILS_ROOT=/path/to/repo

set -euo pipefail

REPO_ROOT="${ML4ILS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
LEGACY_DIR="$REPO_ROOT/code/pylib"

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <data_dir> <num_procs>" >&2
  exit 1
fi

DATA_DIR="$1"
NUM_PROCS="$2"

echo "processing dir: ${DATA_DIR}..."
cd "$LEGACY_DIR"
PYTHONWARNINGS=ignore python train_and_test_new.py --procs "$NUM_PROCS" \
  --data "$DATA_DIR" \
  --predict \
  --classify \
  --folds 10
# --config "$DATA_DIR/config.json"
