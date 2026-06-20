#!/usr/bin/env bash
# Scale run: FULL fine-tune (no LoRA cap) on a big single GPU (80GB), then the
# same in-process eval. Knobs via env so we can iterate without editing.
#   N_TRAIN, MIN_ELO, EPOCHS, BSZ, GACC, MAXPLIES, LR
# Data prep is skipped if data/train.jsonl already exists, so a re-launch after
# an OOM retrains WITHOUT re-streaming the (slow) Lichess pull.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/_scratch
export PYTHONUNBUFFERED=1          # live logs incl. the eval debug dump (no block-buffer)
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq stockfish >/dev/null 2>&1
export STOCKFISH_PATH="$(command -v stockfish || echo /usr/games/stockfish)"
echo "stockfish: $STOCKFISH_PATH"

pip install -q --upgrade pip
pip install -q unsloth trl peft bitsandbytes datasets transformers accelerate python-chess python-dotenv

N_TRAIN=${N_TRAIN:-80000}; MIN_ELO=${MIN_ELO:-1800}; EPOCHS=${EPOCHS:-3}
BSZ=${BSZ:-8}; GACC=${GACC:-4}; MAXPLIES=${MAXPLIES:-100}; LR=${LR:-1e-5}

if [ -s data/train.jsonl ]; then
  echo "### data prep SKIPPED (data/train.jsonl exists: $(wc -l < data/train.jsonl) games)"
else
  echo "### data prep (N=$N_TRAIN, elo>=$MIN_ELO, max_plies=$MAXPLIES)"
  python data/prepare.py --n-train-games "$N_TRAIN" --n-val-games 1000 --n-eval-games 300 \
    --min-elo "$MIN_ELO" --max-plies "$MAXPLIES" 2>&1 | tee results/_scratch/data.log
fi

echo "### FULL fine-tune (epochs=$EPOCHS bsz=$BSZ gacc=$GACC lr=$LR) + merge"
python train/finetune.py --full --merge --epochs "$EPOCHS" \
  --batch-size "$BSZ" --grad-accum "$GACC" --lr "$LR" 2>&1 | tee results/_scratch/train.log

echo "### in-process eval (base vs full-FT, vs Stockfish)"
python eval/run_local.py --debug 10 2>&1 | tee results/_scratch/eval.log
echo "### RUN DONE"
