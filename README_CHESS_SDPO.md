# Chess SDPO (Qwen3-8B Reasoning)

Train `Qwen/Qwen3-8B` with SDPO while using `reasoning_trace` as privileged teacher-only context.

Main config: `verl/trainer/config/chess_sdpo.yaml`.

## Dataset schema

Each row should include:

- `fen`: FEN string.
- `valid_moves`: list of legal UCI moves (recommended).
- `reasoning_trace`: natural-language analysis/hint text.
- `chosen_move`: target best move in UCI.

## Prepare data

Recommended (directly from Hugging Face):

```bash
python data/preprocess_chess.py \
  --load_from_hf amazingvince/chess-traces \
  --output_dir datasets/chess \
  --train_size 100000 \
  --test_size 100 \
  --max_reasoning_chars 4000
```

With local files:

```bash
python data/preprocess_chess.py \
  --train_path datasets/chess/train.jsonl \
  --test_path datasets/chess/test.jsonl \
  --output_dir datasets/chess \
  --max_reasoning_chars 3000
```

Optional:

- Hide legal moves from the student prompt: add `--no_legal_moves`.
- Limit legal moves shown: add `--max_legal_moves 96`.
- Set a different reasoning cap: add `--max_reasoning_chars <N>`.

## Train

Preflight environment check:

```bash
python scripts/check_torch_cuda.py --min-gpus 2
```

```bash
export SDPO_DIR=/path/to/SDPO
export TASK=datasets/chess
export EXPERIMENT=chess-sdpo-qwen3
export MODEL_PATH=Qwen/Qwen3-8B

bash training/verl_training.sh "$EXPERIMENT" chess_sdpo "$TASK"
```

## How privileged hints are used

1. Student prompt contains only chess state context (`fen` and optional `valid_moves`), not `reasoning_trace`.
2. `reasoning_trace` is stored in `extra_info` and appended into reward feedback in `verl/utils/reward_score/feedback/chess.py`.
3. SDPO reprompting injects that feedback into the teacher message via `feedback_template`.
4. Teacher logits (with privileged hint context) are distilled back to the student policy.

This follows the same teacher/student asymmetry pattern as OPSD-style privileged teacher prompting.

`max_reasoning_chars=3000` is intentionally conservative: in a 1000-trace sample from
`amazingvince/chess-traces`, we observed `avg_chars=834.26`, `p95=1568`, and `max=2117`.

## Important config notes

- `actor_rollout_ref.model.use_fused_kernels` is set to `False`.
  SDPO top-k/full-logit distillation requires logits that fused-kernel mode does not expose in this implementation.
- Qwen3 thinking mode is enabled via `data.apply_chat_template_kwargs.enable_thinking: true`.
- `remove_thinking_from_demonstration: true` prevents directly copying `<think>...</think>` spans from demonstrations.
- Chess reward shaping now gives small credit for strict single-move formatting and legal-but-wrong moves,
  while keeping only correct moves above `success_reward_threshold`.

## Qwen3 sampling defaults

The config uses the Qwen3 thinking defaults:

- `temperature: 0.6`
- `top_p: 0.95`
- `top_k: 20`

## Practical troubleshooting

- If W&B init fails with `403 permission denied`:
  - Re-login with a valid write-capable key:

```bash
wandb login --relogin <your_wandb_api_key>
```

  - Set your own entity explicitly (or leave it unset to use your default account):

```bash
export WANDB_ENTITY=<your_wandb_username_or_team>
export WANDB_PROJECT=SDPO-chess
```

- If rollout/actor log-prob drift is high (`training/rollout_probs_diff_mean > 0.01` on long reasoning runs), add:

```bash
+actor_rollout_ref.rollout.engine_kwargs.vllm.disable_cascade_attn=True
```

- If vLLM OOM appears during wake-up/resume (`wake_up(tags=["kv_cache"])`) and you want to keep context length unchanged, lower rollout concurrency/cache reservation instead:

```bash
export GPU_MEM_UTIL=0.6
export MAX_NUM_BATCHED_TOKENS=4096
export MAX_NUM_SEQS=32
```

  For persistent OOM during `wake_up(tags=["kv_cache"])`, use a stricter profile:

```bash
export GPU_MEM_UTIL=0.5
export MAX_NUM_BATCHED_TOKENS=2048
export MAX_NUM_SEQS=16
```

- If outputs are too long, lower `data.max_response_length` or disable thinking mode.

## Upstream fork note

Your fork currently includes chess-specific files (`chess_sdpo.yaml`, `preprocess_chess.py`, chess reward).
Recent `lasgroup/SDPO` upstream commits focus on reproducibility/dependency pinning and do not include this chess extension.
