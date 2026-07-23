"""Regression tests recomputing the README's headline claims from the committed
per-position result rows. These pin the July 23 corrections: CPL is conditional
on legality (different denominators per model), and the matched-position
comparison — the fair one — must keep showing the improvement. Stdlib only.
"""

import json
import statistics
from pathlib import Path

RESULTS = Path(__file__).resolve().parent.parent / "results" / "uci_fullft"


def _rows(name):
    with open(RESULTS / f"rows__{name}.jsonl") as f:
        return [json.loads(line) for line in f]


def test_legality_counts_match_readme():
    base, ft = _rows("chess-base-uci"), _rows("chess-ft-uci")
    assert len(base) == len(ft) == 400
    assert sum(r["legal"] for r in base) == 180  # 45.0%
    assert sum(r["legal"] for r in ft) == 392    # 98.0%


def test_conditional_medians_match_readme():
    for name, med in (("chess-base-uci", 255), ("chess-ft-uci", 55)):
        legal = [r["acpl"] for r in _rows(name) if r["acpl"] is not None]
        assert statistics.median(legal) == med


def test_matched_position_medians():
    # the fair comparison: only positions where BOTH models played a legal move
    base = {r["id"]: r["acpl"] for r in _rows("chess-base-uci") if r["acpl"] is not None}
    ft = {r["id"]: r["acpl"] for r in _rows("chess-ft-uci") if r["acpl"] is not None}
    shared = sorted(set(base) & set(ft))
    assert len(shared) == 179
    assert statistics.median(base[i] for i in shared) == 252
    assert statistics.median(ft[i] for i in shared) == 47
