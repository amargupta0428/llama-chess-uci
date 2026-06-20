"""
Unified model-move interface for every contestant.

  vllm:<name>     base / fine-tuned Llama, via vLLM /v1/completions (PGN continuation)
  openai:gpt-4o   frontier baseline, via chat (asked for the next move in SAN)

ask_move(model_spec, history_san) -> raw model text; the caller parses + legality-
checks it against the board (a model can output an illegal/garbage move — that's a
measured outcome, not an error).
"""
import os

import chess

try:                                  # dotenv is only needed for the API backends;
    from dotenv import load_dotenv    # the in-process eval (run_local.py) uses only
    load_dotenv()                     # parse_move, so never hard-fail if it's absent.
except ModuleNotFoundError:
    pass

_clients: dict[str, object] = {}


def _openai(base_url: str | None):
    from openai import OpenAI
    key = "vllm" if base_url else "openai"
    if key not in _clients:
        _clients[key] = (OpenAI(base_url=base_url, api_key="EMPTY", max_retries=6)
                         if base_url else OpenAI(max_retries=6))
    return _clients[key]


def ask_move(model_spec: str, history_san: str) -> str:
    """Return the model's raw next-move text (not yet validated)."""
    backend, _, model = model_spec.partition(":")
    if backend == "vllm":
        c = _openai(os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1"))
        r = c.completions.create(
            model=model, prompt=history_san + " ", max_tokens=8,
            temperature=0.0, stop=["\n"],
        )
        return r.choices[0].text.strip()
    if backend == "openai":
        c = _openai(None)
        r = c.chat.completions.create(
            model=model, max_tokens=8, temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a chess engine. Given the moves "
                 "so far in SAN, reply with ONLY the next move in standard algebraic "
                 "notation (e.g. Nf3, exd5, O-O). No move number, no commentary."},
                {"role": "user", "content": history_san},
            ],
        )
        return (r.choices[0].message.content or "").strip()
    raise ValueError(f"unknown backend: {model_spec!r}")


def parse_move(raw: str, board: chess.Board, fmt: str = "san"):
    """First real move token of `raw` -> legal chess.Move, or None. `fmt` = san|uci."""
    if fmt == "uci":
        return parse_move_uci(raw, board)
    toks = raw.strip().split()
    if not toks:
        return None
    if toks[0].rstrip(".").isdigit():              # drop a standalone move-number token
        toks = toks[1:]
    if not toks:
        return None
    tok = toks[0].lstrip("0123456789.").strip()    # drop a glued move number
    tok = tok.rstrip("+#!?").strip()               # drop check/annotation marks
    if not tok:
        return None
    try:
        return board.parse_san(tok)
    except (ValueError, chess.IllegalMoveError, chess.InvalidMoveError, chess.AmbiguousMoveError):
        return None


def parse_move_uci(raw: str, board: chess.Board):
    """First UCI token of `raw` (e.g. 'e2e4', 'e7e8q') -> legal chess.Move, or None."""
    toks = raw.strip().split()
    if not toks:
        return None
    tok = toks[0].strip().rstrip("+#!?")           # UCI carries no marks, but be tolerant
    try:
        mv = chess.Move.from_uci(tok)
    except (ValueError, chess.InvalidMoveError):
        return None
    return mv if board.is_legal(mv) else None
