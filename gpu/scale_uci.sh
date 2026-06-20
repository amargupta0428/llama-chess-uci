#!/usr/bin/env bash
# UCI scale run: ~400k games, FULL fine-tune (all 8B weights) + sequence packing.
# UCI won the format A/B (95.3 vs 89.7 legal); this is the definitive UCI model on
# much more data to see if legality climbs toward competitive (~99%). Needs an 80GB
# GPU (gradient checkpointing + KV-cache off + lr 1e-5 are set by the --full path).
# Packing concatenates the short movetext into full windows -> ~7x fewer steps.
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/_scratch
export PYTHONUNBUFFERED=1
apt-get update -qq >/dev/null 2>&1 && apt-get install -y -qq stockfish >/dev/null 2>&1
export STOCKFISH_PATH="$(command -v stockfish || echo /usr/games/stockfish)"
echo "stockfish: $STOCKFISH_PATH"
pip install -q --upgrade pip
pip install -q unsloth trl peft bitsandbytes datasets transformers accelerate python-chess python-dotenv

N=${N_TRAIN:-400000}; ELO=${MIN_ELO:-1600}; EP=${EPOCHS:-2}
BSZ=${BSZ:-8}; GACC=${GACC:-4}; MAXP=${MAXPLIES:-100}; LR=${LR:-1e-5}
EVAL_LIMIT=${EVAL_LIMIT:-400}; DEPTH=${DEPTH:-10}; GAMES=${GAMES:-10}

if [ -s data/train_uci.jsonl ]; then
  echo "### data prep SKIPPED ($(wc -l < data/train_uci.jsonl) games present)"
else
  echo "### data prep (N=$N, elo>=$ELO, both formats) — this streams a lot, be patient"
  python data/prepare.py --n-train-games "$N" --n-val-games 1000 --n-eval-games 300 \
    --min-elo "$ELO" --max-plies "$MAXP" 2>&1 | tee results/_scratch/data.log
fi

# Packing is OFF by default: for short chess movetext, packing forces full 1024-token
# windows (~9s/it) while plain dynamic padding of the short sequences runs ~1.8s/it —
# packing was a net SLOWDOWN here. Set PACK=1 to re-enable. --save-steps keeps ONE
# rolling 32GB checkpoint in outputs/uci so a stop+relaunch resumes (disk must survive;
# a hard-terminated pod wipes it, so credits + death-detection are the backstop).
PACK_FLAG=""; [ "${PACK:-0}" = "1" ] && PACK_FLAG="--packing"
HF_FLAG=""; [ -n "${HF_REPO:-}" ] && HF_FLAG="--hf-repo $HF_REPO"   # push merged model to HF
echo "### FULL fine-tune UCI ${PACK_FLAG:-(no packing)} (${EP}ep, bsz=$BSZ gacc=$GACC lr=$LR) + merge ${HF_FLAG}"
python train/finetune.py --full --merge --train-file data/train_uci.jsonl --out-name chess-ft-uci \
  $PACK_FLAG $HF_FLAG --epochs "$EP" --batch-size "$BSZ" --grad-accum "$GACC" --lr "$LR" \
  --save-steps "${SAVE_STEPS:-800}" --out-dir outputs/uci \
  2>&1 | tee results/_scratch/train_uci.log

echo "### EVAL UCI (base vs ft, vs Stockfish ${GAMES} games)"
python eval/run_local.py --format uci --ft merged-model/chess-ft-uci \
  --base-name chess-base-uci --ft-name chess-ft-uci \
  --limit "$EVAL_LIMIT" --depth "$DEPTH" --games "$GAMES" --debug 10 2>&1 | tee results/_scratch/eval_uci.log
echo "### RUN DONE"
