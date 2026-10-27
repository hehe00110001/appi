#!/bin/bash
#SBATCH --job-name=appi-splitcmp
#SBATCH --partition=A6000
#SBATCH --gres=gpu:1
#SBATCH --array=0-3%2
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=/home/weizg/wei/fangyh/appi/logs/splitcmp-%A_%a.log
set -euo pipefail
cd /home/weizg/wei/fangyh/appi
unset PYTHONPATH PYTHONHOME TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
index="${SLURM_ARRAY_TASK_ID:-0}"
if ! [[ "$index" =~ ^[0-3]$ ]]; then exit 1; fi
variant=original
data=data/processed/joint_training_v2
cache=data/processed/residue_cache_original_v1.pt
if (( index >= 2 )); then
    variant=complex40
    data=data/processed/joint_training_complex40_v1
    cache=data/processed/residue_cache_complex40_v1.pt
fi
mode=absolute
lambda_diff=0
if (( index % 2 == 1 )); then mode=joint; lambda_diff=0.3; fi
output="runs/affinity_split_comparison_v1/${variant}/${mode}_seed42"
resume=()
if [[ -f "$output/last.pt" ]]; then resume=(--resume); fi
/home/weizg/wei/miniconda3/envs/appi-ppiformer/bin/python -u scripts/train_residue_affinity.py \
    --data "$data" --cache "$cache" --out "$output" --mode "$mode" --lambda-diff "$lambda_diff" \
    --pair-absolute-weight 1 --seed 42 --device cuda --threads "$OMP_NUM_THREADS" \
    --width 128 --layers 2 --heads 4 --batch-size 64 --microbatch-size 8 --attention-limit 1000000 "${resume[@]}"
