#!/bin/bash
#SBATCH --partition=A100
#SBATCH --exclude=gpu03
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --job-name=appi-egfr-seeds43-44
#SBATCH --time=04:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/adaptyv-egfr-seeds43-44-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python
new_out=outputs/adaptyv_egfr_seed43_44_v1
merged_out=outputs/adaptyv_egfr_multiseed_v1
test ! -e "$new_out/metrics.json"
"$py" -u - <<'PY'
import os
import socket
import torch
free, total = torch.cuda.mem_get_info()
print(f'GPU preflight: host={socket.gethostname()}, CUDA_VISIBLE_DEVICES={os.environ.get("CUDA_VISIBLE_DEVICES")}, device={torch.cuda.get_device_name()}, free={free / 2**30:.2f} GiB, total={total / 2**30:.2f} GiB', flush=True)
if free < 2 * 2**30:
    raise RuntimeError('Allocated GPU has less than 2 GiB free before inference; check shared GPU occupancy.')
PY
"$py" -u scripts/evaluate_adaptyv_egfr.py --device cuda --seeds 43 44 --out "$new_out"
"$py" -u scripts/summarize_adaptyv_egfr_multiseed.py --inputs outputs/adaptyv_egfr_v1 "$new_out" --out "$merged_out"
"$py" -u scripts/report_adaptyv_egfr.py --out "$merged_out"
