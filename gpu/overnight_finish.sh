#!/usr/bin/env bash
# Overnight tail: wait for the UCI full-FT run to finish, then preserve the merged
# model to the LOCAL Mac and only THEN tear down (preserve_and_teardown.py gates the
# terminate on a verified ~16GB local file). Run under `caffeinate` so the Mac stays
# awake long enough to do the download. Reads the live pod from results/_scratch/pod.json.
set -uo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
J="import json;print(json.load(open('results/_scratch/pod.json'))"
HOST=$(python -c "$J['ssh_host'])"); PORT=$(python -c "$J['ssh_port'])"); PID=$(python -c "$J['pod_id'])")
echo "overnight: watching $HOST:$PORT pod=$PID"

for i in $(seq 1 160); do                      # up to ~13 hr
  sleep 300
  ALIVE=$(python -c "import runpod,os;from dotenv import load_dotenv;load_dotenv('.env',override=True);runpod.api_key=os.environ['RUNPOD_API_KEY'];p=runpod.get_pod('$PID') or {};print('A' if p.get('desiredStatus')=='RUNNING' else 'D')" 2>/dev/null)
  if [ "$ALIVE" != "A" ]; then echo "[$i] *** POD DIED ($ALIVE) — cannot preserve (terminate wipes disk). ALERT ***"; exit 2; fi
  OUT=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR -o ConnectTimeout=20 -p "$PORT" root@"$HOST" \
    'cd /root/Chess && grep -ac "RUN DONE" uci.log; grep -acE "Traceback|out of memory|iostream error" uci.log; grep -aoE "[0-9]+/12500 \[|### [A-Za-z].*" uci.log | tail -n1' 2>/dev/null)
  DONE=$(echo "$OUT" | sed -n 1p); ERR=$(echo "$OUT" | sed -n 2p); MARK=$(echo "$OUT" | sed -n 3p)
  echo "[$i ~$((i*5))min] '$MARK' done=$DONE err=$ERR"
  if [ "$DONE" = "1" ]; then echo "RUN DONE -> preserving model locally before teardown"; break; fi
  if [ "$ERR" -ge 1 ] 2>/dev/null; then echo "[$i] RUN ERROR in log — leaving pod UP for inspection"; exit 3; fi
done

# Download + verify (>=15GB) + terminate ONLY if verified. Local-only (no HF arg).
python gpu/preserve_and_teardown.py
echo "OVERNIGHT COMPLETE"
