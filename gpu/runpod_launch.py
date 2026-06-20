"""
Create / inspect / terminate a RunPod pod for chess training + eval.

  python gpu/runpod_launch.py create     # spin up GPU, wait for SSH, save pod.json
  python gpu/runpod_launch.py status
  python gpu/runpod_launch.py terminate  # destroy the pod (ALWAYS run when done)

GPU type via env GPU_TYPE (default "NVIDIA A40" for the spike; use e.g.
"NVIDIA A100 80GB PCIe" or "NVIDIA H100 PCIe" for full fine-tune).
"""
import json
import os
import pathlib
import sys
import time

import runpod
from dotenv import load_dotenv

load_dotenv(".env", override=True)
runpod.api_key = os.environ["RUNPOD_API_KEY"]

SCRATCH = pathlib.Path("results/_scratch")
SCRATCH.mkdir(parents=True, exist_ok=True)
POD_FILE = SCRATCH / "pod.json"
IMAGE = "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
GPU_TYPE = os.environ.get("GPU_TYPE", "NVIDIA A40")
DISK_GB = int(os.environ.get("DISK_GB", "80"))


def create():
    pub = pathlib.Path(os.path.expanduser("~/.ssh/id_ed25519.pub")).read_text().strip()
    pod = runpod.create_pod(
        name="chess-finetune", image_name=IMAGE, gpu_type_id=GPU_TYPE,
        cloud_type="SECURE", gpu_count=int(os.environ.get("GPU_COUNT", "1")),
        container_disk_in_gb=DISK_GB, volume_in_gb=0,
        ports="22/tcp,8000/http", support_public_ip=True, start_ssh=True,
        env={"PUBLIC_KEY": pub},
    )
    pid = pod["id"]
    print(f"created pod {pid} ({GPU_TYPE})")
    # If SSH never comes up (or we're interrupted), TERMINATE the pod before giving
    # up — otherwise a multi-GPU fallback loop overwrites pod.json and orphans this
    # pod, which keeps billing forever. (Bit us with a stray L40S once.)
    try:
        for i in range(120):
            time.sleep(6)
            p = runpod.get_pod(pid) or {}
            ports = (p.get("runtime") or {}).get("ports") or []
            ssh = next((x for x in ports if x.get("privatePort") == 22 and x.get("isIpPublic")), None)
            print(f"  [{i}] status={p.get('desiredStatus')} ports={'yes' if ports else 'no'}")
            if ssh:
                POD_FILE.write_text(json.dumps(
                    {"pod_id": pid, "ssh_host": ssh["ip"], "ssh_port": ssh["publicPort"]}, indent=2))
                print(f"SSH_READY {ssh['ip']}:{ssh['publicPort']}")
                return
        print(f"TIMEOUT waiting for SSH — terminating {pid} to avoid an orphan")
        runpod.terminate_pod(pid)
    except BaseException:
        try:
            runpod.terminate_pod(pid)
            print(f"cleaned up pod {pid} after error/interrupt")
        except Exception:
            pass
        raise


def status():
    p = runpod.get_pod(json.loads(POD_FILE.read_text())["pod_id"]) or {}
    print(json.dumps({"status": p.get("desiredStatus"), "running": bool(p.get("runtime"))}, indent=2))


def terminate():
    pid = json.loads(POD_FILE.read_text())["pod_id"]
    runpod.terminate_pod(pid)
    print("terminated", pid)


if __name__ == "__main__":
    {"create": create, "status": status, "terminate": terminate}[sys.argv[1]]()
