#!/usr/bin/env bash
# Single clean run: data -> LoRA train -> IN-PROCESS eval (no vLLM). One env.
#   bash gpu/run.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/_scratch
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq stockfish >/dev/null 2>&1
export STOCKFISH_PATH="$(command -v stockfish || echo /usr/games/stockfish)"
echo "stockfish: $STOCKFISH_PATH"

pip install -q --upgrade pip
pip install -q unsloth trl peft bitsandbytes datasets transformers accelerate python-chess python-dotenv

echo "### data prep"
python data/prepare.py --n-train-games 30000 --n-val-games 500 --n-eval-games 200 \
  2>&1 | tee results/_scratch/data.log
echo "### train (LoRA) + merge"
python train/finetune.py --merge 2>&1 | tee results/_scratch/train.log
echo "### in-process eval (base vs fine-tuned, vs Stockfish)"
# --debug 10 prints prompt -> raw gen -> parse (no-space vs trailing-space) for the
# first 10 positions of each model BEFORE scoring, so the log SHOWS what the model
# emits. Base Llama should now climb to a believable legal-move rate; if it doesn't,
# the debug dump tells us exactly why instead of leaving us trusting a bogus number.
python eval/run_local.py --debug 10 2>&1 | tee results/_scratch/eval.log
echo "### RUN DONE"
