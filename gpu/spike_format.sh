#!/usr/bin/env bash
# CONTROLLED FORMAT A/B: same games, one LoRA per notation (SAN vs UCI), eval both.
# Cheap test of the hypothesis "UCI cuts illegal moves" BEFORE betting a big full-FT
# on it. Legal-move rate is the metric that decides; eval is capped/shallow to stay
# cheap (format signal shows clearly at this scale). Knobs via env.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/_scratch
export PYTHONUNBUFFERED=1
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq stockfish >/dev/null 2>&1
export STOCKFISH_PATH="$(command -v stockfish || echo /usr/games/stockfish)"
echo "stockfish: $STOCKFISH_PATH"
pip install -q --upgrade pip
pip install -q unsloth trl peft bitsandbytes datasets transformers accelerate python-chess python-dotenv

N=${N_TRAIN:-40000}; ELO=${MIN_ELO:-1800}; EP=${EPOCHS:-1}; MAXP=${MAXPLIES:-100}
EVAL_LIMIT=${EVAL_LIMIT:-300}; DEPTH=${DEPTH:-8}; GAMES=${GAMES:-6}

if [ -s data/train.jsonl ] && [ -s data/train_uci.jsonl ]; then
  echo "### data prep SKIPPED (both formats present)"
else
  echo "### data prep — writes BOTH train.jsonl (SAN) + train_uci.jsonl (UCI), same games"
  python data/prepare.py --n-train-games "$N" --n-val-games 500 --n-eval-games 300 \
    --min-elo "$ELO" --max-plies "$MAXP" 2>&1 | tee results/_scratch/data.log
fi

echo "### TRAIN SAN LoRA"
python train/finetune.py --merge --train-file data/train.jsonl --out-name chess-ft-san \
  --epochs "$EP" 2>&1 | tee results/_scratch/train_san.log
echo "### EVAL SAN"
python eval/run_local.py --format san --ft merged-model/chess-ft-san \
  --base-name chess-base-san --ft-name chess-ft-san \
  --limit "$EVAL_LIMIT" --depth "$DEPTH" --games "$GAMES" --debug 8 2>&1 | tee results/_scratch/eval_san.log

echo "### TRAIN UCI LoRA"
python train/finetune.py --merge --train-file data/train_uci.jsonl --out-name chess-ft-uci \
  --epochs "$EP" 2>&1 | tee results/_scratch/train_uci.log
echo "### EVAL UCI"
python eval/run_local.py --format uci --ft merged-model/chess-ft-uci \
  --base-name chess-base-uci --ft-name chess-ft-uci \
  --limit "$EVAL_LIMIT" --depth "$DEPTH" --games "$GAMES" --debug 8 2>&1 | tee results/_scratch/eval_uci.log

echo "### FORMAT A/B DONE"
