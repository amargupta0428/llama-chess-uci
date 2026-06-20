# Llama-3.1-8B-Chess-UCI

A full fine-tune of Llama 3.1 8B that learns to play chess by predicting the next move in UCI notation. Measured against a Stockfish oracle, the fine-tune takes the base model from **45% to 98% legal moves** and cuts **median centipawn loss from 255 to 55** (4.6x better move quality).

**Built with Llama.**

> **Honest summary up front.** This is an objective, Stockfish-verified fine-tuning result on a metric that is hard to game. It is **not** a model that beats a real chess engine. It still loses **0 of 10** games against a ~1200-rated Stockfish, because 98% legality is not enough to survive a full game and the model still throws occasional blunders. The value here is the measured before/after delta and a reproducible pipeline, not engine-beating strength.

## Results

UCI full fine-tune of Llama 3.1 8B, 400k Lichess games, 1 epoch. Base and fine-tune are evaluated on the **same frozen 400-position set**, with Stockfish as the move-quality oracle.

| Metric | Base | Fine-tuned | Change |
|---|---|---|---|
| Legal-move rate | 45.0% | 98.0% | +53 pts |
| Median centipawn loss (lower is better) | 255 | 55 | 4.6x better |
| Matches Stockfish's top move | 3.8% | 28.0% | ~7x |
| Matches the human move actually played | 4.5% | 36.3% | ~8x |
| Games won vs Stockfish-1200 (out of 10) | 0 | 0 | no change |

Median centipawn loss is the honest strength number. The mean is much higher because of occasional large blunders, which is also why the model still loses full games despite decent typical play.

**Metric definitions.** Legal-move rate is the fraction of generated moves that are legal in the given position. Centipawn loss is how much worse the model's move is than Stockfish's best move in that position, per Stockfish analysis. The match rates are exact-move agreement with Stockfish's top move and with the move the human actually played in the source game.

## What this demonstrates

A controlled, end-to-end fine-tuning project with an objective ground truth. Chess legality and centipawn loss cannot be talked up: a move is legal or it is not, and Stockfish scores quality independently of the model. The result is a clean, measured delta (45% to 98% legal, 4.6x lower median centipawn loss) produced by a reproducible pipeline, with the limitations stated honestly rather than hidden.

## Method

- **Task.** Given a game prefix, predict the next move in UCI notation. Completion-style fine-tune.
- **Base model.** `unsloth/Meta-Llama-3.1-8B` (Llama 3.1 8B base, not Instruct). Exact revision pinned in `results/uci_fullft/manifest.json`.
- **Data.** Lichess standard games via Hugging Face, filtered to a minimum Elo of 1600, 400k games, seed 42. Games are streamed and converted into move-prediction examples. The raw training file is **not committed** (too large); it is regenerated deterministically by `data/prepare.py` from the same seed.
- **Training.** Full fine-tune (not LoRA), 1 epoch, learning rate 1e-5, effective batch size 32, max sequence length 1024, cosine schedule, `adamw_8bit`, seed 42, on a single H100. Full hyperparameters live in the manifest.
- **Eval.** `eval/run_local.py` loads the merged model and scores legal-move rate, centipawn loss, and move-match rates over a frozen position set, using Stockfish (`eval/engine.py`) as the oracle, plus 10 full games against Stockfish capped near 1200 Elo.

## Why UCI and a full fine-tune: the experiments

The final configuration was not a guess. It was chosen by a short sequence of controlled experiments.

| Experiment | Setup | Fine-tuned legal-move rate |
|---|---|---|
| Spike | SAN, LoRA, 30k games | 91.6% |
| Scale | SAN, full fine-tune, 80k games | 93.7% |
| Format A/B | SAN LoRA vs UCI LoRA, matched | SAN 89.7% vs UCI 95.3% |
| Final | UCI, full fine-tune, 400k games | 98.0% |

Two findings drove the final choice. First, SAN fine-tunes plateaued around 94% legal regardless of data size or LoRA-vs-full. Second, a matched A/B showed a UCI fine-tune reaches higher legality than a SAN one (95.3% vs 89.7% as LoRA). Switching the target format to UCI, then scaling the data and moving to a full fine-tune, broke the SAN plateau and reached 98%.

**One honest detail on the baseline.** The base model's legal-move rate is format-dependent. It scores around 88% legal when prompted in SAN but only 45% in UCI, because UCI is a less familiar notation to the base model. The 45% to 98% headline is the within-UCI delta. The reason UCI is the right target is not that it makes the base look worse, it is that the fine-tuned UCI model beats the fine-tuned SAN model head to head in the A/B above.

## Reproducibility

- Everything is pinned by seed 42 and the manifest (`results/uci_fullft/manifest.json`): dataset, min Elo, game counts, base model and revision, training hyperparameters, and the sha256 of the frozen eval set.
- The training data regenerates deterministically from `data/prepare.py`; it is not shipped because of size, which is standard practice.
- The merged weights are regenerable from the committed config and seed (full fine-tune, seed 42, pinned base revision). The 16GB model is not committed to the repo and is available on request.

## Repo structure

```
data/prepare.py        Streams Lichess games into UCI training data + frozen eval set + manifest
train/finetune.py      Completion-style fine-tune (LoRA or --full), merge, optional HF push
eval/run_local.py      The eval used: loads the model, scores legality / centipawn loss / match
eval/engine.py         Stockfish oracle (best move, centipawn loss)
eval/common.py         Move parsing (SAN + UCI)
gpu/                    RunPod orchestration: launch, train, preserve weights, guarded teardown
results/               Eval outputs for every experiment, incl. 800 per-position rows for the final run
```

## Limitations

- **Does not beat a real engine.** 0 of 10 versus Stockfish-1200. At 98% legality, roughly half of full games still end on a single illegal move before strength matters, and the model still blunders in the games it finishes.
- **Headline baseline is UCI-format.** See the note above. The cross-format comparison is stated openly rather than buried.
- **Strength is below 1200 Elo.** Median centipawn loss of 55 is decent per move, but the mean (much higher) shows the tail of blunders that loses games.

## License

Built with Llama. This is a derivative of Meta Llama 3.1 8B, used and redistributed under the Llama 3.1 Community License (see `LICENSE`) and the Acceptable Use Policy (see `USE_POLICY.md`). Llama 3.1 is licensed under the Llama 3.1 Community License, Copyright Meta Platforms, Inc. Model name: Llama-3.1-8B-Chess-UCI.
