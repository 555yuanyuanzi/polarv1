#!/usr/bin/env bash
set -euo pipefail

TORCH_VARIANT="${1:-cpu}"
PYTHON_VERSION="${2:-3.11}"

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT_DIR}"

echo "Creating virtual environment with uv..."
uv venv .venv --python "${PYTHON_VERSION}"

PYTHON_EXE="${ROOT_DIR}/.venv/bin/python"
if [[ ! -x "${PYTHON_EXE}" ]]; then
    echo "Python executable not found at ${PYTHON_EXE}" >&2
    exit 1
fi

echo "Installing PyTorch and torchvision for variant: ${TORCH_VARIANT}"
case "${TORCH_VARIANT}" in
    cpu)
        uv pip install --python "${PYTHON_EXE}" torch torchvision
        ;;
    cu121)
        uv pip install --python "${PYTHON_EXE}" --index-url https://download.pytorch.org/whl/cu121 torch torchvision
        ;;
    cu124)
        uv pip install --python "${PYTHON_EXE}" --index-url https://download.pytorch.org/whl/cu124 torch torchvision
        ;;
    *)
        echo "Unsupported TORCH_VARIANT: ${TORCH_VARIANT}" >&2
        echo "Use one of: cpu, cu121, cu124" >&2
        exit 1
        ;;
esac

echo "Installing project dependencies from requirements.txt"
uv pip install --python "${PYTHON_EXE}" -r requirements.txt

echo
echo "Environment ready."
echo "Activate with:"
echo "  source .venv/bin/activate"
echo "Then edit configs/gopro_v1.yaml and run:"
echo "  python train.py --config configs/debug_v1.yaml"
