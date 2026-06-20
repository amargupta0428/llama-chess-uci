"""
Stockfish oracle via python-chess UCI. Provides the objective ground truth:
best move, position eval, and centipawn loss of a played move. Also a
skill-capped engine for fair head-to-head opponents.
"""
import atexit
import os
import shutil

import chess
import chess.engine

# Track every engine we open and ALWAYS quit them on exit — including on
# exceptions and Ctrl-C (atexit runs for normal exit + KeyboardInterrupt).
# This prevents the orphaned-Stockfish leak that can pile up across runs and
# exhaust memory. (atexit can't catch SIGKILL/hard crash — so also keep memory
# per engine tiny, below.)
_OPENED: list[chess.engine.SimpleEngine] = []


@atexit.register
def _close_all() -> None:
    for e in _OPENED:
        try:
            e.quit()
        except Exception:
            pass


def find_stockfish() -> str:
    p = os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
    if not p:
        raise RuntimeError("Stockfish not found. `brew install stockfish` / "
                           "`apt-get install -y stockfish`, or set STOCKFISH_PATH.")
    return p


def open_engine(elo: int | None = None, hash_mb: int = 16,
                threads: int = 1) -> chess.engine.SimpleEngine:
    """Full-strength oracle by default; pass elo to cap strength for opponents.

    Hash and threads are capped low so each process stays ~lean (a few hundred MB
    max), and every engine is registered for guaranteed cleanup on exit.
    """
    eng = chess.engine.SimpleEngine.popen_uci(find_stockfish())
    _OPENED.append(eng)
    cfg = {"Threads": threads, "Hash": hash_mb}
    if elo is not None:
        cfg.update({"UCI_LimitStrength": True, "UCI_Elo": int(elo)})
    try:
        eng.configure(cfg)
    except chess.engine.EngineError:
        try:                                  # retry without the elo cap if unsupported
            eng.configure({"Threads": threads, "Hash": hash_mb})
        except chess.engine.EngineError:
            pass
    return eng


def close_all() -> None:
    """Explicit cleanup for callers that want to free engines mid-run."""
    _close_all()
    _OPENED.clear()


def _cp_pov(score: chess.engine.PovScore, color: chess.Color) -> int:
    """Centipawns from `color`'s POV; mate mapped to a large finite value."""
    return score.pov(color).score(mate_score=100000)


def best_move(engine, board: chess.Board, depth: int = 12) -> chess.Move:
    return engine.play(board, chess.engine.Limit(depth=depth)).move


def centipawn_loss(engine, board: chess.Board, move: chess.Move, depth: int = 12) -> int:
    """How much worse `move` is than Stockfish's best, in centipawns (>=0).

    Evaluated from the moving side's POV. 0 = matched the engine's choice.
    """
    mover = board.turn
    best = _cp_pov(engine.analyse(board, chess.engine.Limit(depth=depth))["score"], mover)
    board.push(move)
    after = _cp_pov(engine.analyse(board, chess.engine.Limit(depth=depth))["score"], mover)
    board.pop()
    return max(0, best - after)
