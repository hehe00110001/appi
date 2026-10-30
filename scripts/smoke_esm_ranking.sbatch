#!/bin/bash
#SBATCH --partition=A6000
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --job-name=appi-esm-smoke
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/esm-smoke-%j.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 CUBLAS_WORKSPACE_CONFIG=:4096:8
py=/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python
"$py" -m pytest -q tests/test_esm_residue.py
"$py" -u scripts/extract_esm2_features.py --smoke-only
for mode in seq esm_graph esm_nogeom fusion; do
    for arm in B R; do
        "$py" -u scripts/train_esm_ranking.py --out "runs/esm2_smoke_v1/${mode}_${arm}_seed42" \
            --mode "$mode" --arm "$arm" --limit 8 --epoch-records 8 --epochs 2 --patience 2 --device cuda \
            --esm-cache data/processed/esm2_complex40_v1/features_smoke.pt
    done
done
"$py" scripts/verify_esm_ranking_smoke.py
