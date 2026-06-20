#!/usr/bin/env bash
# Turnkey feasibility SPIKE on the RunPod box: data -> LoRA fine-tune -> serve ->
# objective eval (base vs fine-tuned) -> verdict. Runs entirely on the pod.
#
#   bash gpu/spike.sh
#
# Decision gate: does the fine-tune clearly beat base on legal-move rate / ACPL /
# move-match, and hold up vs a weak Stockfish? If yes -> scale (full FT, big GPU).
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/_scratch

echo "### 1/6 system deps (stockfish for the oracle, ninja for vLLM kernels)"
apt-get update -qq >/dev/null 2>&1
apt-get install -y -qq stockfish ninja-build cmake >/dev/null 2>&1
export STOCKFISH_PATH="$(command -v stockfish || echo /usr/games/stockfish)"
echo "stockfish: $STOCKFISH_PATH"

echo "### 2/6 training deps + data prep (high-Elo PGN -> move sequences)"
pip install -q --upgrade pip
pip install -q unsloth trl peft bitsandbytes datasets transformers accelerate python-chess
python data/prepare.py --n-train-games 20000 --n-val-games 300 --n-eval-games 150 \
  2>&1 | tee results/_scratch/data.log

echo "### 3/6 LoRA fine-tune (spike) + merge for serving"
python train/finetune.py --merge 2>&1 | tee results/_scratch/train.log

echo "### 4/6 isolated serve/eval venv (vLLM + chess eval)"
python -m venv /opt/serve
/opt/serve/bin/pip install -q --upgrade pip
# Pin the trio for this pod's CUDA-12.4 driver: vLLM 0.6.3 (torch 2.4.0+cu121,
# driver-safe) + matching transformers (0.6.3 breaks on newer transformers:
# "TokenizersBackend has no attribute all_special_tokens_extended").
/opt/serve/bin/pip install -q "vllm==0.6.3.post1" "transformers==4.45.2" openai python-chess python-dotenv
SERVE_PY=/opt/serve/bin/python

serve_and_eval () {   # $1 = base|ft   $2 = served-name
  echo ">>> serving $1 as $2"
  VLLM_PY="$SERVE_PY" bash serve/vllm_serve.sh "$1" > "results/_scratch/vllm_$2.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && break; sleep 3; done
  echo ">>> $2 up — move accuracy"
  STOCKFISH_PATH="$STOCKFISH_PATH" "$SERVE_PY" eval/move_accuracy.py --model "vllm:$2" --concurrency 16
  if [ "$2" = "chess-ft" ]; then
    echo ">>> ft vs weak Stockfish (10 games)"
    STOCKFISH_PATH="$STOCKFISH_PATH" "$SERVE_PY" eval/play_games.py \
      --white "vllm:chess-ft" --black "stockfish:1200" --games 10
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 4
}

echo "### 5/6 eval BASE then FINE-TUNED"
serve_and_eval base chess-base
serve_and_eval ft   chess-ft

echo "### 6/6 spike verdict"
"$SERVE_PY" - <<'PY'
import json, pathlib
R = pathlib.Path("results")
def load(n):
    p = R / n
    return json.loads(p.read_text()) if p.exists() else {}
b = load("moveacc__vllm__chess-base.json"); f = load("moveacc__vllm__chess-ft.json")
print("\n================ SPIKE VERDICT ================")
print(f"{'metric':<22}{'base':>10}{'fine-tuned':>14}")
for k, label in [("legal_move_rate","legal-move rate"),
                 ("move_match_actual","match human move"),
                 ("move_match_stockfish","match SF best"),
                 ("mean_acpl_legal","ACPL (lower=better)")]:
    print(f"{label:<22}{str(b.get(k)):>10}{str(f.get(k)):>14}")
print("===============================================")
PY
echo "### SPIKE DONE — copy results/ back, then decide on scaling"
