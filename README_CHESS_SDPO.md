# Chess SDPO (Qwen3-8B)

Train Qwen3-8B with SDPO using privileged chess reasoning traces as teacher hints.

This setup:
1. Uses the student prompt (FEN + optional legal moves).
2. Feeds the teacher a reprompt that includes a successful solution (if any) plus
   a feedback section that always contains `reasoning_trace`.
3. Distills the teacher logits back into the student policy.

The config lives at `verl/trainer/config/chess_sdpo.yaml`.

## Dataset

Expected columns per row:

- `fen`: FEN string.
- `valid_moves`: list of legal moves in UCI (optional but recommended).
- `reasoning_trace`: natural-language hint / analysis text.
- `chosen_move`: Stockfish best move in UCI.

Dataset repo ID: `amazingvince/chess-traces`.

## Install

Follow `INSTALL.md` in the repo root.

You will need a CUDA-capable GPU and a recent PyTorch build compatible with your driver.

## Prepare data

The preprocessing script can load the dataset directly from HuggingFace and automatically limit the dataset size. By default, it limits to 100k training samples and 100 test samples.

**Recommended approach** (loads directly from HuggingFace):

```bash
python data/preprocess_chess.py \
  --load_from_hf amazingvince/chess-traces \
  --output_dir datasets/chess
```

To customize the dataset sizes:

```bash
python data/preprocess_chess.py \
  --load_from_hf amazingvince/chess-traces \
  --output_dir datasets/chess \
  --train_size 50000 \
  --test_size 200
```

If you prefer to hide legal moves from the student prompt:

```bash
python data/preprocess_chess.py \
  --load_from_hf amazingvince/chess-traces \
  --output_dir datasets/chess \
  --no_legal_moves
```

**Alternative approach** (using local JSONL files):

If you have already downloaded the dataset to local files:

```bash
python data/preprocess_chess.py \
  --train_path datasets/chess/train.jsonl \
  --test_path datasets/chess/test.jsonl \
  --output_dir datasets/chess
```

Note: Size limiting (via `--train_size` and `--test_size`) is also applied when using local files.

## Train

```bash
export SDPO_DIR=/path/to/SDPO
export TASK=datasets/chess
export EXPERIMENT=chess-sdpo-qwen3

bash training/verl_training.sh $EXPERIMENT chess_sdpo $TASK
```

## Reasoning handling (important)

This setup is designed for privileged teacher hints:

1. The student prompt **does not** include `reasoning_trace`.
2. The reward function in `verl/utils/reward_score/feedback/chess.py` always appends
   `reasoning_trace` to the feedback text.
3. SDPO’s reprompt template includes that feedback, so the **teacher** sees the hint.
4. `apply_chat_template_kwargs.enable_thinking: true` allows Qwen3 to emit `<think>...</think>`
   blocks during rollouts.
5. `remove_thinking_from_demonstration: true` strips `<think>` blocks from successful
   demonstrations to avoid distilling chain-of-thought into the student’s visible output.

If you want to distill chain-of-thought, set:

- `actor_rollout_ref.actor.self_distillation.remove_thinking_from_demonstration: false`
- Consider updating the student prompt to explicitly allow reasoning output.

## Qwen3 sampling defaults

Qwen3 recommends thinking-mode sampling with:

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`

The chess config already sets these defaults for both rollout and validation sampling:

- `temperature=0.6`
- `top_p=0.95`
- `top_k=20`

## Troubleshooting

If the dataset is private or gated, login first:

```bash
huggingface-cli login
```

If you see excessively long outputs, lower `data.max_response_length` or disable thinking mode
with `data.apply_chat_template_kwargs.enable_thinking: false`.
