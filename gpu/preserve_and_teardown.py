"""
SAVE-BEFORE-TEARDOWN guard. Runs LOCALLY after the pod's run finishes.

  python gpu/preserve_and_teardown.py            # local-only (download + verify)
  python gpu/preserve_and_teardown.py <hf_repo>  # also require the HF copy

1. scp the merged fp16 model off the pod to ./models/chess-ft-uci  (local backup)
2. verify the LOCAL copy exists at full size (>= ~15GB for an 8B fp16 model)
3. (optional) verify the HF repo holds the weights
4. TEARDOWN GUARD: terminate the pod ONLY if every requested check passes;
   otherwise leave the pod UP and exit non-zero so the weights can be recovered.
"""
import json
import os
import pathlib
import subprocess
import sys

import runpod
from dotenv import load_dotenv

load_dotenv(".env", override=True)
runpod.api_key = os.environ["RUNPOD_API_KEY"]

REPO = sys.argv[1] if len(sys.argv) > 1 else None       # None => local-only
LOCAL = pathlib.Path("models/chess-ft-uci")
MIN_BYTES = 14 * 1024**3       # 14 GiB floor. The fp16 8B merge is ~16.06 decimal GB
                               # = ~14.96 GiB; a 15-GiB floor (16.1 decimal GB) wrongly
                               # rejects the FULL model. A truncated scp loses a whole
                               # shard (~12GB or less), which 14 GiB still catches.
SSH = ["-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR"]

pod = json.load(open("results/_scratch/pod.json"))
host, port, pid = pod["ssh_host"], pod["ssh_port"], pod["pod_id"]

# 1) pull the merged model down ------------------------------------------------
LOCAL.mkdir(parents=True, exist_ok=True)
print(f"[1/4] scp merged model pod -> {LOCAL} (~16GB, a few min) ...")
subprocess.run(["scp", "-r", *SSH, "-P", str(port),
                f"root@{host}:/root/Chess/merged-model/chess-ft-uci/.", str(LOCAL)], check=True)

# 2) verify LOCAL size ---------------------------------------------------------
local_bytes = sum(f.stat().st_size for f in LOCAL.rglob("*") if f.is_file())
local_ok = local_bytes >= MIN_BYTES
print(f"[2/4] LOCAL  {local_bytes/1e9:.1f} GB at {LOCAL.resolve()} -> {'OK' if local_ok else 'TOO SMALL'}")
for f in sorted(LOCAL.rglob("*")):
    if f.is_file():
        print(f"        {f.relative_to(LOCAL)}  {f.stat().st_size/1e9:.2f} GB")

# 3) optional HF check ---------------------------------------------------------
if REPO:
    from huggingface_hub import HfApi
    from transformers import AutoConfig
    try:
        files = HfApi(token=os.environ.get("HF_TOKEN")).list_repo_files(REPO)
        AutoConfig.from_pretrained(REPO, token=os.environ.get("HF_TOKEN"))
        hf_ok = any(f.endswith(".safetensors") for f in files) and "config.json" in files
    except Exception as e:
        hf_ok = False
        print("    HF check error:", e)
    print(f"[3/4] HF     {REPO} -> {'OK' if hf_ok else 'FAIL'}")
else:
    hf_ok = True
    print("[3/4] HF     skipped (local-only mode)")

# 4) TEARDOWN GUARD ------------------------------------------------------------
if local_ok and hf_ok:
    runpod.terminate_pod(pid)
    print(f"[4/4] SAVE VERIFIED -> terminated pod {pid}")
    print(f"\nLOCAL MODEL: {LOCAL.resolve()}")
    if REPO:
        print(f"HF:          https://huggingface.co/{REPO}")
else:
    print(f"[4/4] NOT TERMINATING (local_ok={local_ok} hf_ok={hf_ok}). Pod {pid} left UP for recovery.")
    sys.exit(1)
