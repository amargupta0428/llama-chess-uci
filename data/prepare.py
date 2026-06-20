"""
Build the chess training + eval data from high-Elo Lichess games.

Streams `Lichess/standard-chess-games` (HF), filters to high-Elo games, parses
each with python-chess (guarantees legal, clean SAN), and writes:

  train.jsonl / val.jsonl   {"text": "1. e4 e5 2. Nf3 ..."}  (completion-style,
                            next-move prediction over clean SAN move sequences)
  eval_positions.jsonl      frozen mid-game positions for objective scoring:
                            {game_id, ply, history_san, fen, actual_move_san/uci}
  manifest.json             counts + sha256 of the frozen eval set (reproducible)

Run on the GPU box for the big pull (fast network); locally for small samples.
  python data/prepare.py --n-train-games 30000 --n-eval-games 300 --min-elo 2000
"""
import argparse
import hashlib
import io
import json
import random
from pathlib import Path

import chess
import chess.pgn

HERE = Path(__file__).resolve().parent
SEED = 42
DATASET = "Lichess/standard-chess-games"


def parse_movetext(movetext: str):
    """movetext str -> (list[SAN moves], list[(ply, fen_before, san, uci)]) or None."""
    game = chess.pgn.read_game(io.StringIO(movetext))
    if game is None:
        return None
    board = game.board()
    sans, steps = [], []
    for ply, move in enumerate(game.mainline_moves()):
        if not board.is_legal(move):
            break
        fen_before = board.fen()
        san = board.san(move)
        sans.append(san)
        steps.append((ply, fen_before, san, move.uci()))
        board.push(move)
    return sans, steps


def movetext_from_sans(sans: list[str]) -> str:
    """Clean SAN list -> '1. e4 e5 2. Nf3 ...' (standard movetext, no annotations)."""
    out = []
    for i, san in enumerate(sans):
        if i % 2 == 0:
            out.append(f"{i // 2 + 1}.")
        out.append(san)
    return " ".join(out)


def history_prefix(sans: list[str], ply: int) -> str:
    """Movetext up to (not including) the move at `ply` — the model's input."""
    prefix = movetext_from_sans(sans[:ply])
    # cue the side to move: add the move number if white is to move next
    if ply % 2 == 0:
        prefix = (prefix + f" {ply // 2 + 1}.").strip()
    return prefix


def movetext_from_ucis(ucis: list[str]) -> str:
    """UCI move list -> 'e2e4 e7e5 g1f3 ...' (space-joined, no move numbers).

    UCI encodes a move as from-square+to-square (e.g. e2e4, e7e8q for promotion),
    so there is no SAN grammar (disambiguation, captures, checks, castling spelling)
    for the model to get wrong — the whole point of testing this format.
    """
    return " ".join(ucis)


def history_prefix_uci(ucis: list[str], ply: int) -> str:
    """UCI move list up to (not including) `ply` — the model's input (no cue needed)."""
    return movetext_from_ucis(ucis[:ply])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train-games", type=int, default=30000)
    ap.add_argument("--n-val-games", type=int, default=500)
    ap.add_argument("--n-eval-games", type=int, default=300,
                    help="held-out games sampled for frozen eval positions")
    ap.add_argument("--min-elo", type=int, default=2000)
    ap.add_argument("--max-plies", type=int, default=80, help="truncate long games")
    ap.add_argument("--min-plies", type=int, default=20, help="skip very short games")
    ap.add_argument("--eval-plies", type=int, nargs="*", default=[12, 20, 30, 40],
                    help="plies to snapshot as eval positions (skips opening book)")
    args = ap.parse_args()
    random.seed(SEED)

    from datasets import load_dataset
    ds = load_dataset(DATASET, split="train", streaming=True)

    need = args.n_train_games + args.n_val_games + args.n_eval_games
    kept, streamed = [], 0
    print(f"Streaming {DATASET}, filtering Elo>={args.min_elo} for {need} games ...")
    for row in ds:
        streamed += 1
        try:
            if int(row["WhiteElo"]) < args.min_elo or int(row["BlackElo"]) < args.min_elo:
                continue
        except (TypeError, ValueError):
            continue
        parsed = parse_movetext(row["movetext"])
        if not parsed:
            continue
        sans, steps = parsed
        if len(sans) < args.min_plies:
            continue
        kept.append((sans[: args.max_plies], steps))
        if len(kept) >= need:
            break
        if len(kept) % 2000 == 0:
            print(f"  kept {len(kept)}/{need} (streamed {streamed})")
    print(f"Done streaming: kept {len(kept)} from {streamed} games")

    random.shuffle(kept)
    train = kept[: args.n_train_games]
    val = kept[args.n_train_games: args.n_train_games + args.n_val_games]
    evalg = kept[args.n_train_games + args.n_val_games:]

    def write_games(base, games):
        """Write the SAME games in BOTH formats: <base>.jsonl (SAN) + <base>_uci.jsonl."""
        san_p, uci_p = HERE / f"{base}.jsonl", HERE / f"{base}_uci.jsonl"
        with open(san_p, "w") as fs, open(uci_p, "w") as fu:
            for sans, steps in games:
                ucis = [s[3] for s in steps][: len(sans)]   # match the truncated SAN length
                fs.write(json.dumps({"text": movetext_from_sans(sans)}) + "\n")
                fu.write(json.dumps({"text": movetext_from_ucis(ucis)}) + "\n")
        print(f"  wrote {len(games):>6} games -> {san_p.name} + {uci_p.name}")

    write_games("train", train)
    write_games("val", val)

    # frozen eval positions: snapshot chosen plies from held-out games (both formats)
    eval_rows = []
    for gid, (sans, steps) in enumerate(evalg):
        ucis = [s[3] for s in steps]
        for ply, fen, san, uci in steps:
            if ply in args.eval_plies and ply < len(sans):
                eval_rows.append({
                    "game_id": gid, "ply": ply,
                    "history_san": history_prefix(sans, ply),
                    "history_uci": history_prefix_uci(ucis, ply),
                    "fen": fen, "actual_move_san": san, "actual_move_uci": uci,
                })
    with open(HERE / "eval_positions.jsonl", "w") as f:
        for r in eval_rows:
            f.write(json.dumps(r) + "\n")
    print(f"  wrote {len(eval_rows):>6} eval positions -> eval_positions.jsonl")

    content = (HERE / "eval_positions.jsonl").read_text()
    (HERE / "manifest.json").write_text(json.dumps({
        "dataset": DATASET, "min_elo": args.min_elo, "seed": SEED,
        "n_train_games": len(train), "n_val_games": len(val),
        "n_eval_games": len(evalg), "n_eval_positions": len(eval_rows),
        "eval_sha256": hashlib.sha256(content.encode()).hexdigest(),
    }, indent=2))
    print(f"  manifest.json written (eval sha256 "
          f"{hashlib.sha256(content.encode()).hexdigest()[:12]}...)")


if __name__ == "__main__":
    main()
