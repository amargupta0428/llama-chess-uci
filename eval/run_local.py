"""
In-process chess eval — NO vLLM (which kept failing on RunPod driver/version
hell). Loads each model with transformers in the same env that trains, generates
next moves directly, and scores objectively against Stockfish.

Evaluates BASE and FINE-TUNED in one process and prints the verdict.

  python eval/run_local.py --base unsloth/Meta-Llama-3.1-8B --ft merged-model/chess-ft

Outputs results/moveacc__<name>.json + results/games__chess-ft_vs_stockfish.json
"""
import argparse
import json
import math
import statistics
from pathlib import Path

import chess
import chess.engine
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from common import parse_move
from engine import open_engine, best_move, centipawn_loss

ROOT = Path(__file__).resolve().parent.parent
DATA, RESULTS = ROOT / "data", ROOT / "results"
RESULTS.mkdir(exist_ok=True)


def load(path):
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    return model, tok


def gen(model, tok, prompt, max_new=8):
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=max_new, do_sample=False,
                             pad_token_id=tok.pad_token_id)
    return tok.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=True)


def hist_of(p, fmt):
    return p["history_uci"] if fmt == "uci" else p["history_san"]


def debug_examples(model, tok, name, positions, n=10, fmt="san"):
    """Print prompt -> raw gen -> parse verdict so we SEE what the model emits.

    Training movetext carries a LEADING space on every move (' Nf3' / ' g1f3'),
    so the model learned that token; a TRAILING space in the prompt pushes it off
    that boundary (the trailing-whitespace bug). Shows both variants side by side.
    """
    print(f"\n----- DEBUG {name} [{fmt}]: prompt/gen/parse on first {n} positions -----")
    for p in positions[:n]:
        board = chess.Board(p["fen"])
        hist = hist_of(p, fmt)
        raw_nospace = gen(model, tok, hist)            # the fix: no trailing space
        raw_space = gen(model, tok, hist + " ")        # the old (buggy) variant
        mv_n = parse_move(raw_nospace, board, fmt)
        mv_s = parse_move(raw_space, board, fmt)
        print(f"  hist(tail) ...{hist[-40:]!r}  actual={p['actual_move_uci']!r}")
        print(f"    no-space : raw={raw_nospace!r:<28} -> {mv_n.uci() if mv_n else 'ILLEGAL/None'}")
        print(f"    w/ space : raw={raw_space!r:<28} -> {mv_s.uci() if mv_s else 'ILLEGAL/None'}")
    print("-------------------------------------------------------------\n")


def move_accuracy(model, tok, name, eng, positions, depth=12, fmt="san"):
    legal_losses, rows = [], []
    for i, p in enumerate(positions, 1):
        board = chess.Board(p["fen"])
        mv = parse_move(gen(model, tok, hist_of(p, fmt)), board, fmt)   # no trailing space
        legal = mv is not None
        match_a = legal and mv.uci() == p["actual_move_uci"]
        match_sf, acpl = False, None
        if legal:
            sf = best_move(eng, board, depth); match_sf = mv == sf
            acpl = centipawn_loss(eng, board, mv, depth); legal_losses.append(acpl)
        rows.append({"id": f"{p['game_id']}.{p['ply']}", "legal": legal, "match_actual": match_a,
                     "match_sf": match_sf, "acpl": acpl})
        if i % 50 == 0:
            print(f"  {name}: {i}/{len(positions)}")
    # Per-position rows persisted (per-example pattern) so ACPL can be re-derived
    # honestly offline — median, and ACPL on the subset where BOTH models are legal
    # (the mean-over-legal is confounded when models have different legal sets).
    with open(RESULTS / f"rows__{name}.jsonl", "w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    n = len(rows)
    s = {"model": name, "n": n, "n_legal": len(legal_losses),
         "legal_move_rate": round(sum(r["legal"] for r in rows) / n, 4),
         "move_match_actual": round(sum(r["match_actual"] for r in rows) / n, 4),
         "move_match_stockfish": round(sum(r["match_sf"] for r in rows) / n, 4),
         "mean_acpl_legal": round(sum(legal_losses) / len(legal_losses), 1) if legal_losses else None,
         "median_acpl_legal": round(statistics.median(legal_losses), 1) if legal_losses else None}
    (RESULTS / f"moveacc__{name}.json").write_text(json.dumps(s, indent=2))
    print(f"  -> {s}")
    return s


def _game_prompt(moves, fmt):
    """Build the model's prompt from the move list so far (no trailing space)."""
    if fmt == "uci":
        return " ".join(moves)
    parts = []
    for i, s in enumerate(moves):
        if i % 2 == 0: parts.append(f"{i//2+1}.")
        parts.append(s)
    return (" ".join(parts) + (f" {len(moves)//2+1}." if len(moves) % 2 == 0 else "")).strip()


def play_vs_stockfish(model, tok, eng_elo, n_games=10, max_plies=160, fmt="san", name="chess-ft"):
    eng = open_engine(elo=eng_elo)
    adj = open_engine()
    wins = draws = losses = 0
    for g in range(n_games):
        board, moves = chess.Board(), []        # move strings in `fmt`
        model_white = (g % 2 == 0)
        while not board.is_game_over() and len(moves) < max_plies:
            if (board.turn == chess.WHITE) == model_white:
                mv = parse_move(gen(model, tok, _game_prompt(moves, fmt)), board, fmt)
                if mv is None or not board.is_legal(mv):
                    losses += 1; break
            else:
                mv = eng.play(board, chess.engine.Limit(depth=8)).move
            moves.append(mv.uci() if fmt == "uci" else board.san(mv)); board.push(mv)
        else:
            if board.is_game_over():
                r = board.result()
                won = (r == "1-0") == model_white
                wins += won and r != "1/2-1/2"; draws += r == "1/2-1/2"; losses += (r != "1/2-1/2" and not won)
            else:
                cp = adj.analyse(board, chess.engine.Limit(depth=12))["score"].white().score(mate_score=100000)
                cp = cp if model_white else -cp
                wins += cp > 100; losses += cp < -100; draws += -100 <= cp <= 100
    eng.quit(); adj.quit()
    score = (wins + 0.5 * draws) / n_games
    out = {"opponent": f"stockfish:{eng_elo}", "games": n_games, "wins": wins,
           "draws": draws, "losses": losses, "score_rate": round(score, 3)}
    if 0 < score < 1:
        out["est_elo"] = round(eng_elo - 400 * math.log10(1 / score - 1))
    (RESULTS / f"games__{name}_vs_stockfish.json").write_text(json.dumps(out, indent=2))
    print(f"  -> {out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="unsloth/Meta-Llama-3.1-8B")
    ap.add_argument("--ft", default=str(ROOT / "merged-model/chess-ft"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--games", type=int, default=10)
    ap.add_argument("--opp-elo", type=int, default=1200)
    ap.add_argument("--debug", type=int, default=0,
                    help="print prompt/gen/parse for the first N positions of each model")
    ap.add_argument("--debug-only", action="store_true",
                    help="run only the debug dump (no Stockfish scoring) — cheap sanity check")
    ap.add_argument("--format", choices=["san", "uci"], default="san",
                    help="move notation the model was trained on (picks history field + parser)")
    ap.add_argument("--base-name", default="chess-base")
    ap.add_argument("--ft-name", default="chess-ft")
    args = ap.parse_args()
    fmt = args.format

    positions = [json.loads(l) for l in (DATA / "eval_positions.jsonl").read_text().splitlines() if l.strip()]
    if args.limit:
        positions = positions[: args.limit]

    if args.debug_only:
        print(f"== DEBUG-ONLY [{fmt}]: BASE ({args.base}) ==")
        m, t = load(args.base); debug_examples(m, t, args.base_name, positions, args.debug or 10, fmt)
        del m; torch.cuda.empty_cache()
        print(f"== DEBUG-ONLY [{fmt}]: FINE-TUNED ({args.ft}) ==")
        m, t = load(args.ft); debug_examples(m, t, args.ft_name, positions, args.debug or 10, fmt)
        return

    eng = open_engine()

    print(f"== BASE [{fmt}] ({args.base}) ==")
    m, t = load(args.base)
    if args.debug:
        debug_examples(m, t, args.base_name, positions, args.debug, fmt)
    b = move_accuracy(m, t, args.base_name, eng, positions, args.depth, fmt)
    del m; torch.cuda.empty_cache()

    print(f"== FINE-TUNED [{fmt}] ({args.ft}) ==")
    m, t = load(args.ft)
    if args.debug:
        debug_examples(m, t, args.ft_name, positions, args.debug, fmt)
    f = move_accuracy(m, t, args.ft_name, eng, positions, args.depth, fmt)
    eng.quit()
    if args.games:
        print(f"== FT vs Stockfish {args.opp_elo} ({args.games} games) ==")
        play_vs_stockfish(m, t, args.opp_elo, args.games, fmt=fmt, name=args.ft_name)

    print("\n================ VERDICT ================")
    print(f"{'metric':<20}{'base':>10}{'fine-tuned':>14}")
    for k, lbl in [("legal_move_rate", "legal-move %"), ("move_match_actual", "match human"),
                   ("move_match_stockfish", "match SF"), ("mean_acpl_legal", "ACPL (lower=better)")]:
        print(f"{lbl:<20}{str(b[k]):>10}{str(f[k]):>14}")
    print("========================================")


if __name__ == "__main__":
    main()
