#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "Usage: bash scripts/train_multi_gpu_conda.sh <conda_env_name> [config_path]"
  exit 1
fi

ENV_NAME="$1"
CONFIG_PATH="${2:-configs/gopro_fbeb.yaml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda was not found in PATH."
  echo "Initialize conda in your shell first, then rerun this script."
  exit 1
fi

CONDA_BASE="$(conda info --base)"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"

python scripts/launch_train.py --config "${CONFIG_PATH}"
