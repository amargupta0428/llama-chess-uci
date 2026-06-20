#!/usr/bin/env bash
# Eval-only re-entry (training/data/merged model already done). Rebuilds the
# serve venv pinned to a vLLM whose torch matches the pod's CUDA-12.4 driver
# (vLLM 0.23 pulled a too-new torch -> "driver too old"). Then serves base + FT
# and runs the objective eval.
#
#   bash gpu/eval_only.sh
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p results/_scratch
export STOCKFISH_PATH="$(command -v stockfish || echo /usr/games/stockfish)"

echo "### serve venv pinned to torch-2.4-compatible vLLM"
rm -rf /opt/serve
python -m venv /opt/serve
/opt/serve/bin/pip install -q --upgrade pip
/opt/serve/bin/pip install -q "vllm==0.6.3.post1" openai python-chess python-dotenv
SERVE_PY=/opt/serve/bin/python
"$SERVE_PY" -c "import torch; print('serve torch', torch.__version__, 'cuda ok:', torch.cuda.is_available())"

serve_and_eval () {   # $1 = base|ft   $2 = served-name
  echo ">>> serving $1 as $2"
  VLLM_PY="$SERVE_PY" bash serve/vllm_serve.sh "$1" > "results/_scratch/vllm_$2.log" 2>&1 &
  local pid=$!
  for i in $(seq 1 240); do curl -sf http://localhost:8000/health >/dev/null 2>&1 && break; sleep 3; done
  if ! curl -sf http://localhost:8000/health >/dev/null 2>&1; then
    echo "!! $2 server NOT healthy — see results/_scratch/vllm_$2.log"; kill "$pid" 2>/dev/null; return
  fi
  echo ">>> $2 healthy — move accuracy"
  STOCKFISH_PATH="$STOCKFISH_PATH" "$SERVE_PY" eval/move_accuracy.py --model "vllm:$2" --concurrency 16
  if [ "$2" = "chess-ft" ]; then
    echo ">>> ft vs weak Stockfish (10 games)"
    STOCKFISH_PATH="$STOCKFISH_PATH" "$SERVE_PY" eval/play_games.py \
      --white "vllm:chess-ft" --black "stockfish:1200" --games 10
  fi
  kill "$pid" 2>/dev/null; wait "$pid" 2>/dev/null; sleep 4
}

serve_and_eval base chess-base
serve_and_eval ft   chess-ft

"$SERVE_PY" results/charts.py 2>/dev/null || true
"$SERVE_PY" - <<'PY'
import json, pathlib
R = pathlib.Path("results")
def load(n):
    p = R / n; return json.loads(p.read_text()) if p.exists() else {}
b = load("moveacc__vllm__chess-base.json"); f = load("moveacc__vllm__chess-ft.json")
print("\n================ SPIKE VERDICT ================")
print(f"{'metric':<22}{'base':>10}{'fine-tuned':>14}")
for k,label in [("legal_move_rate","legal-move rate"),("move_match_actual","match human"),
                ("move_match_stockfish","match SF best"),("mean_acpl_legal","ACPL (lower=better)")]:
    print(f"{label:<22}{str(b.get(k)):>10}{str(f.get(k)):>14}")
print("===============================================")
PY
echo "### EVAL DONE"
