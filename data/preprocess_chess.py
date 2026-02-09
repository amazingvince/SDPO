import argparse
import json
import os
from typing import Iterable

import datasets
import pyarrow as pa
import pyarrow.parquet as pq


DEFAULT_SYSTEM_PROMPT = (
    "You are a strong chess engine. Given a position in FEN, choose the best move. "
    "Respond with a single UCI move and nothing else."
)


def _to_large(field: pa.Field) -> pa.Field:
    t = field.type
    if pa.types.is_string(t):
        return pa.field(field.name, pa.large_string(), field.nullable, field.metadata)
    if pa.types.is_binary(t):
        return pa.field(field.name, pa.large_binary(), field.nullable, field.metadata)
    if pa.types.is_list(t):
        return pa.field(
            field.name,
            pa.large_list(_to_large(pa.field("item", t.value_type)).type),
            field.nullable,
            field.metadata,
        )
    if pa.types.is_struct(t):
        return pa.field(
            field.name,
            pa.struct([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in t]),
            field.nullable,
            field.metadata,
        )
    return field


def _large_schema(schema: pa.Schema) -> pa.Schema:
    return pa.schema([_to_large(pa.field(f.name, f.type, f.nullable, f.metadata)) for f in schema])


def write_rowgrouped_large(ds, path: str, rows_per_group: int = 32):
    """Cast to LargeString/LargeList and write many small row groups."""
    tbl: pa.Table = ds.data.table
    tbl = tbl.cast(_large_schema(tbl.schema))
    n = len(tbl)
    writer = None
    try:
        for start in range(0, n, rows_per_group):
            chunk = tbl.slice(start, min(rows_per_group, n - start))
            if writer is None:
                writer = pq.ParquetWriter(path, chunk.schema, compression="zstd")
            writer.write_table(chunk)
    finally:
        if writer is not None:
            writer.close()


def _load_dataset(path: str) -> datasets.Dataset:
    ext = os.path.splitext(path)[1].lower()
    if ext in [".json", ".jsonl"]:
        return datasets.load_dataset("json", data_files=path, split="train")
    if ext in [".parquet"]:
        return datasets.load_dataset("parquet", data_files=path, split="train")
    if ext in [".csv", ".tsv"]:
        return datasets.load_dataset("csv", data_files=path, split="train")
    raise ValueError(f"Unsupported file extension: {ext}")


def _coerce_moves(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip().lower() for v in value if str(v).strip()]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).strip().lower() for v in parsed if str(v).strip()]
        except json.JSONDecodeError:
            pass
        return [v.lower() for v in value.replace(",", " ").split() if v]
    return [str(value).strip().lower()] if str(value).strip() else []


def _normalize_uci_move(value) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_reasoning_trace(value, max_reasoning_chars: int | None) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if max_reasoning_chars is not None and max_reasoning_chars > 0 and len(text) > max_reasoning_chars:
        return text[:max_reasoning_chars].rstrip()
    return text


def _format_legal_moves(moves: Iterable[str], max_moves: int | None) -> str:
    moves = list(moves)
    if max_moves is not None and max_moves > 0:
        moves = moves[:max_moves]
    return " ".join(moves)


def _build_prompt(fen: str, valid_moves, include_legal_moves: bool, max_legal_moves: int | None) -> str:
    if include_legal_moves:
        moves_str = _format_legal_moves(valid_moves, max_legal_moves)
        return f"FEN: {fen}\nLegal moves: {moves_str}\nReturn only the best move in UCI."
    return f"FEN: {fen}\nReturn only the best move in UCI."


def _map_row(
    example,
    idx,
    system_prompt: str,
    include_legal_moves: bool,
    max_legal_moves: int | None,
    max_reasoning_chars: int | None,
):
    fen = example.get("fen", "")
    valid_moves = _coerce_moves(example.get("valid_moves"))
    reasoning_trace = _normalize_reasoning_trace(example.get("reasoning_trace", ""), max_reasoning_chars)
    chosen_move = _normalize_uci_move(example.get("chosen_move", ""))

    prompt = _build_prompt(fen, valid_moves, include_legal_moves, max_legal_moves)

    return {
        "data_source": "chess",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "reward_model": {"style": "chess", "ground_truth": str(chosen_move)},
        "extra_info": {
            "index": str(idx),
            "fen": fen,
            "valid_moves": valid_moves,
            "reasoning_trace": reasoning_trace,
        },
    }


def _write_split(ds: datasets.Dataset, out_path: str, **map_kwargs):
    mapped = ds.map(lambda ex, idx: _map_row(ex, idx, **map_kwargs), with_indices=True)
    write_rowgrouped_large(mapped, out_path)


def _validate_required_columns(ds: datasets.Dataset, split_name: str):
    required = {"fen", "chosen_move"}
    missing = sorted(required - set(ds.column_names))
    if missing:
        raise ValueError(f"Missing required columns in {split_name}: {missing}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess chess dataset into SDPO-compatible parquet.")
    parser.add_argument("--input_path", type=str, help="Path to full dataset file (json/jsonl/csv/parquet).")
    parser.add_argument("--train_path", type=str, help="Optional path to train split file.")
    parser.add_argument("--test_path", type=str, help="Optional path to test split file.")
    parser.add_argument("--load_from_hf", type=str, help="Load dataset directly from HuggingFace (e.g., 'amazingvince/chess-traces').")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for parquet files.")
    parser.add_argument("--test_ratio", type=float, default=0.001, help="Test split ratio if using input_path or load_from_hf.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting.")
    parser.add_argument("--train_size", type=int, default=100_000, help="Maximum number of training samples (applied after loading).")
    parser.add_argument("--test_size", type=int, default=100, help="Maximum number of test samples (applied after loading).")
    parser.add_argument("--system_prompt", type=str, default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--no_legal_moves", action="store_true", help="Do not include legal moves in the prompt.")
    parser.add_argument("--max_legal_moves", type=int, default=None, help="Max number of legal moves to include.")
    parser.add_argument(
        "--max_reasoning_chars",
        type=int,
        default=3000,
        help="Maximum number of characters retained from reasoning_trace for teacher-only hints.",
    )
    args = parser.parse_args()

    if args.load_from_hf:
        # Load directly from HuggingFace
        ds = datasets.load_dataset(args.load_from_hf)
        if "train" in ds and "test" in ds:
            train_ds = ds["train"]
            test_ds = ds["test"]
        else:
            split = ds["train"].train_test_split(test_size=args.test_ratio, seed=args.seed, shuffle=True)
            train_ds = split["train"]
            test_ds = split["test"]
    elif args.train_path and args.test_path:
        train_ds = _load_dataset(args.train_path)
        test_ds = _load_dataset(args.test_path)
    elif args.input_path:
        full_ds = _load_dataset(args.input_path)
        split = full_ds.train_test_split(test_size=args.test_ratio, seed=args.seed, shuffle=True)
        train_ds = split["train"]
        test_ds = split["test"]
    else:
        raise ValueError("Provide --load_from_hf, --input_path, or both --train_path and --test_path.")

    _validate_required_columns(train_ds, "train split")
    _validate_required_columns(test_ds, "test split")

    # Limit dataset sizes
    train_ds = train_ds.select(range(min(args.train_size, len(train_ds))))
    test_ds = test_ds.select(range(min(args.test_size, len(test_ds))))

    os.makedirs(args.output_dir, exist_ok=True)
    map_kwargs = {
        "system_prompt": args.system_prompt,
        "include_legal_moves": not args.no_legal_moves,
        "max_legal_moves": args.max_legal_moves,
        "max_reasoning_chars": args.max_reasoning_chars,
    }

    _write_split(train_ds, os.path.join(args.output_dir, "train.parquet"), **map_kwargs)
    _write_split(test_ds, os.path.join(args.output_dir, "test.parquet"), **map_kwargs)

    print(f"Wrote {len(train_ds)} train and {len(test_ds)} test samples to {args.output_dir}")


if __name__ == "__main__":
    main()
