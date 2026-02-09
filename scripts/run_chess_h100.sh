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

WANDB_ENTITY="${WANDB_ENTITY:-}"
export WANDB_PROJECT="${WANDB_PROJECT:-SDPO-chess}"
# Use TRAINER_LOGGER=console to skip wandb (e.g. if you get 403 permission denied)
export TRAINER_LOGGER="${TRAINER_LOGGER:-[console,wandb]}"

if [ -n "${WANDB_API_KEY:-}" ]; then
  wandb login --relogin "${WANDB_API_KEY}" || true
else
  echo "WANDB_API_KEY not set. If this is the first run, do: wandb login"
fi

if [ -n "${WANDB_ENTITY}" ]; then
  export WANDB_ENTITY
fi

# Optional: HF auth for gated models/datasets
if [ -n "${HF_TOKEN:-}" ] && command -v huggingface-cli >/dev/null 2>&1; then
  huggingface-cli login --token "${HF_TOKEN}" || true
fi

GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.5}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
VLLM_DISABLE_CASCADE_ATTN="${VLLM_DISABLE_CASCADE_ATTN:-false}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"

echo "Running: ${EXPERIMENT}"
echo "SDPO_DIR=${SDPO_DIR}"
echo "TASK=${TASK}"
if [ -n "${WANDB_ENTITY}" ]; then
  echo "WANDB_ENTITY=${WANDB_ENTITY}"
else
  echo "WANDB_ENTITY=<wandb default account>"
fi
echo "WANDB_PROJECT=${WANDB_PROJECT}"
echo "MODEL_PATH=${MODEL_PATH}"
echo "VLLM_DISABLE_CASCADE_ATTN=${VLLM_DISABLE_CASCADE_ATTN}"
echo "MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS}"
echo "MAX_NUM_SEQS=${MAX_NUM_SEQS}"

bash "${ROOT_DIR}/training/verl_training.sh" "${EXPERIMENT}" chess_sdpo "${TASK}" \
  trainer.n_gpus_per_node=2 \
  "trainer.logger=${TRAINER_LOGGER}" \
  trainer.project_name="${WANDB_PROJECT}" \
  trainer.experiment_name="${EXPERIMENT}" \
  trainer.group_name=chess \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${MAX_NUM_BATCHED_TOKENS}" \
  actor_rollout_ref.rollout.max_num_seqs="${MAX_NUM_SEQS}" \
  +actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=${VLLM_DISABLE_CASCADE_ATTN}
