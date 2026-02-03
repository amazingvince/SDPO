#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

echo "SDPO repo: ${ROOT_DIR}"

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPU:"
  nvidia-smi -L || true
else
  echo "Warning: nvidia-smi not found. Ensure NVIDIA drivers are installed."
fi

PYTHON_BIN="${PYTHON_BIN:-python3.12}"
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "Warning: ${PYTHON_BIN} not found, falling back to python3."
  PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" -V

if [ ! -d "${VENV_DIR}" ]; then
  echo "Creating venv at ${VENV_DIR}"
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
fi

source "${VENV_DIR}/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# H100 = Hopper, use CUDA 12.4 wheels.
python -m pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

python -m pip install -r "${ROOT_DIR}/requirements.txt"

# Optional: SGLang/vLLM stack (recommended for fast rollout)
if [ "${INSTALL_SGLANG:-1}" -eq 1 ]; then
  python -m pip install -r "${ROOT_DIR}/requirements_sglang.txt"
fi

# Install SDPO in editable mode
python -m pip install -e "${ROOT_DIR}"

# Optional: Flash-Attn build (requires nvcc + build toolchain)
if [ "${INSTALL_FLASH_ATTN:-0}" -eq 1 ]; then
  python -m pip install -r "${ROOT_DIR}/requirements-cuda.txt" --no-build-isolation
fi

# W&B (if not already installed by requirements)
python -m pip install wandb

echo "Setup complete."
echo "Next: activate venv and run scripts/run_chess_h100.sh"
