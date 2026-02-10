import json
import re
from typing import Iterable


_UCI_RE = re.compile(r"\b([a-h][1-8][a-h][1-8][qrbn]?)\b", re.IGNORECASE)
_STRICT_UCI_RE = re.compile(r"^\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*$", re.IGNORECASE)

# Dense shaping rewards to reduce "yapping" and encourage proper final formatting.
# Keep these below self_distillation.success_reward_threshold (default 0.5 in chess config),
# so only truly correct answers are used as successful demonstrations.
_REWARD_CORRECT = 1.0
_REWARD_LEGAL_STRICT_FORMAT = 0.2
_REWARD_LEGAL_LOOSE_FORMAT = 0.1
_REWARD_ILLEGAL_STRICT_FORMAT = 0.05


def _extract_uci_move(text: str) -> str | None:
    if not text:
        return None
    # Use the last mentioned move as the prediction in non-strict outputs.
    matches = _UCI_RE.findall(text)
    if not matches:
        return None
    return matches[-1].lower()


def _extract_answer_segment(text: str) -> tuple[str, bool]:
    """Extract post-thinking answer segment.

    Returns:
        answer_segment, unclosed_think
    """
    if not text:
        return "", False

    text = str(text)
    if "<think>" in text:
        close_idx = text.rfind("</think>")
        if close_idx == -1:
            return "", True
        answer = text[close_idx + len("</think>") :]
        return answer.strip(), False
    return text.strip(), False


def _is_strict_single_uci(text: str) -> bool:
    if not text:
        return False
    return _STRICT_UCI_RE.match(text) is not None


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
        dict with score, acc, pred, incorrect_format, feedback.

        Score shaping:
            - 1.0: correct best move
            - 0.2: legal move with strict single-UCI format
            - 0.1: legal move with loose/non-strict format
            - 0.05: strict single-UCI format but illegal move
            - 0.0: no valid move / unclosed thinking / other invalid formats
    """
    extra_info = extra_info or {}
    valid_moves = _coerce_moves(extra_info.get("valid_moves"))
    reasoning_trace = extra_info.get("reasoning_trace", "")
    if reasoning_trace is None:
        reasoning_trace = ""
    reasoning_trace = str(reasoning_trace).strip()

    answer_segment, unclosed_think = _extract_answer_segment(solution_str)
    strict_single_uci = _is_strict_single_uci(answer_segment)
    pred_move = _extract_uci_move(answer_segment)
    incorrect_format = 0 if strict_single_uci else 1

    base_feedback_parts: list[str] = []
    if unclosed_think:
        base_feedback_parts.append("Unclosed <think> block. Close with </think> and then output exactly one UCI move.")

    if pred_move is None:
        base_feedback_parts.append("No valid UCI move found. Expected format like e2e4.")
        is_legal = False
        is_correct = False
    else:
        is_legal = True if not valid_moves else pred_move in valid_moves
        is_correct = pred_move == str(ground_truth).lower()
        if not strict_single_uci:
            base_feedback_parts.append("Output format should be exactly one UCI move and nothing else.")

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
    if is_correct:
        reward = _REWARD_CORRECT
    elif is_legal and strict_single_uci:
        reward = _REWARD_LEGAL_STRICT_FORMAT
    elif is_legal:
        reward = _REWARD_LEGAL_LOOSE_FORMAT
    elif strict_single_uci:
        reward = _REWARD_ILLEGAL_STRICT_FORMAT
    else:
        reward = 0.0

    return {
        "score": reward,
        "acc": 1.0 if is_correct else 0.0,
        "pred": pred_move or "",
        "incorrect_format": incorrect_format,
        "strict_single_uci": 1 if strict_single_uci else 0,
        "unclosed_think": 1 if unclosed_think else 0,
        "feedback": feedback,
    }
