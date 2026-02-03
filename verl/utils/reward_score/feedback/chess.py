import json
import re
from typing import Iterable


_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)


def _extract_uci_move(text: str) -> str | None:
    if not text:
        return None
    match = _UCI_RE.search(text)
    if match is None:
        return None
    return match.group(1).lower()


def _coerce_moves(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).lower() for v in value]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        # Try JSON list first
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(v).lower() for v in parsed]
        except json.JSONDecodeError:
            pass
        # Fallback: split on whitespace/commas
        parts = re.split(r"[,\s]+", value)
        return [p.lower() for p in parts if p]
    return [str(value).lower()]


def _format_legal_moves(moves: Iterable[str], max_show: int = 24) -> str:
    moves = list(moves)
    if not moves:
        return ""
    if len(moves) <= max_show:
        return " ".join(moves)
    head = " ".join(moves[:max_show])
    return f"{head} ... (+{len(moves) - max_show} more)"


def compute_score(solution_str: str, ground_truth: str, extra_info: dict | None = None) -> dict:
    """
    Compute score for chess move selection.

    Args:
        solution_str: Model response text.
        ground_truth: Best move in UCI.
        extra_info: Optional dict with keys like:
            - valid_moves: list[str] of legal UCI moves
            - reasoning_trace: privileged hint text
            - fen: position in FEN (for logging/debug)
    Returns:
        dict with score, acc, pred, incorrect_format, feedback
    """
    extra_info = extra_info or {}
    valid_moves = _coerce_moves(extra_info.get("valid_moves"))
    reasoning_trace = extra_info.get("reasoning_trace", "")
    if reasoning_trace is None:
        reasoning_trace = ""
    reasoning_trace = str(reasoning_trace).strip()

    pred_move = _extract_uci_move(solution_str)
    incorrect_format = 0 if pred_move else 1

    base_feedback_parts: list[str] = []
    if pred_move is None:
        base_feedback_parts.append("No valid UCI move found. Expected format like e2e4.")
        is_legal = False
        is_correct = False
    else:
        is_legal = True if not valid_moves else pred_move in valid_moves
        is_correct = pred_move == str(ground_truth).lower()

        if not is_legal:
            legal_preview = _format_legal_moves(valid_moves)
            if legal_preview:
                base_feedback_parts.append(f"Illegal move. Legal moves: {legal_preview}")
            else:
                base_feedback_parts.append("Illegal move.")
        elif not is_correct:
            base_feedback_parts.append("Legal move, but not the best move.")

    feedback_parts: list[str] = []
    if base_feedback_parts:
        feedback_parts.append(" ".join(base_feedback_parts))
    if reasoning_trace:
        feedback_parts.append(reasoning_trace)

    feedback = "\n\n".join(feedback_parts)
    reward = 1.0 if is_correct else 0.0

    return {
        "score": reward,
        "acc": reward,
        "pred": pred_move or "",
        "incorrect_format": incorrect_format,
        "feedback": feedback,
    }
