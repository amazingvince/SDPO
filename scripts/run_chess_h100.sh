#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"

if [ -d "${VENV_DIR}" ]; then
  source "${VENV_DIR}/bin/activate"
fi

export SDPO_DIR="${SDPO_DIR:-$ROOT_DIR}"
export TASK="${TASK:-datasets/chess}"

DATE_TAG="$(date +%Y%m%d_%H%M%S)"
export EXPERIMENT="${EXPERIMENT:-chess-sdpo-h100-${DATE_TAG}}"

export WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-SDPO-chess}"

if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login --relogin "${WANDB_API_KEY}" || true
else
  echo "WANDB_API_KEY not set. If this is the first run, do: wandb login"
fi

# Optional: HF auth for gated models/datasets
if [ -n "${HF_TOKEN:-}" ] && command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli login --token "${HF_TOKEN}" || true
fi

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.7}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"

echo "Running: ${EXPERIMENT}"
echo "SDPO_DIR=${SDPO_DIR}"
echo "TASK=${TASK}"
echo "WANDB_PROJECT=${WANDB_PROJECT}"

bash "${ROOT_DIR}/training/verl_training.sh" "${EXPERIMENT}" chess_sdpo "${TASK}" \
  trainer.n_gpus_per_node=2 \
  trainer.logger=[console,wandb] \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.group_name=chess \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}"
