"""
Build chess result charts + a markdown table from the JSON the eval scripts wrote.
  python results/charts.py
Outputs into results/: chart_legality.png, chart_acpl.png, summary_table.md
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = Path(__file__).resolve().parent
PRETTY = {"vllm:chess-base": "Llama 8B (base)",
          "vllm:chess-ft": "Llama 8B (fine-tuned)",
          "openai:gpt-4o": "GPT-4o"}


def load_moveacc():
    rows = []
    for p in sorted(R.glob("moveacc__*.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def bar(rows, key, title, ylabel, fname, fmt="{:.2f}", pct=False):
    pts = [(PRETTY.get(r["model"], r["model"]), r.get(key)) for r in rows if r.get(key) is not None]
    if not pts:
        print(f"  (skip {fname})"); return
    labels, vals = zip(*pts)
    if pct:
        vals = [v * 100 for v in vals]
    fig, ax = plt.subplots(figsize=(7, 4.2))
    bars = ax.bar(labels, vals)
    ax.set_title(title); ax.set_ylabel(ylabel); ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, fmt.format(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout(); fig.savefig(R / fname, dpi=150); plt.close(fig)
    print(f"  wrote {R / fname}")


def main():
    rows = load_moveacc()
    if not rows:
        print("No moveacc__*.json yet — run eval/move_accuracy.py first."); return
    bar(rows, "legal_move_rate", "Legal-move rate (higher=better)", "%",
        "chart_legality.png", fmt="{:.1f}%", pct=True)
    bar(rows, "mean_acpl_legal", "Avg centipawn loss vs Stockfish (lower=better)",
        "centipawns", "chart_acpl.png", fmt="{:.0f}")

    head = ("| Model | Legal-move % | Match human | Match SF | ACPL (↓) |\n"
            "|---|---|---|---|---|")
    lines = [head]
    for r in sorted(rows, key=lambda r: r.get("mean_acpl_legal") or 9e9):
        lr = f"{r['legal_move_rate']*100:.1f}%" if r.get("legal_move_rate") is not None else "—"
        ma = f"{r['move_match_actual']*100:.1f}%" if r.get("move_match_actual") is not None else "—"
        ms = f"{r['move_match_stockfish']*100:.1f}%" if r.get("move_match_stockfish") is not None else "—"
        ac = r.get("mean_acpl_legal", "—")
        lines.append(f"| {PRETTY.get(r['model'], r['model'])} | {lr} | {ma} | {ms} | {ac} |")
    # append any head-to-head game results
    games = sorted(R.glob("games__*.json"))
    if games:
        lines.append("\n**Head-to-head / vs Stockfish:**")
        for g in games:
            d = json.loads(g.read_text())
            elo = f", est. Elo ~{d['p1_est_elo']}" if "p1_est_elo" in d else ""
            lines.append(f"- {d['p1']} vs {d['p2']}: score {d['p1_score_rate']} "
                         f"({d['p1_wins']}W/{d['draws']}D/{d['p1_losses']}L{elo})")
    (R / "summary_table.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n  wrote {R / 'summary_table.md'}")


if __name__ == "__main__":
    main()
