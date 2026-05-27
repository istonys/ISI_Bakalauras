#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 00:20:00
#SBATCH -o logs/test_gpu_%j.out
#SBATCH -e logs/test_gpu_%j.err

set -e
cd /scratch/lustre/home/$USER/bakalauras_audio_hpc
mkdir -p logs

source .venv/bin/activate

echo "Host: $(hostname)"
echo "Date: $(date)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

nvidia-smi || true

python -m src.check_gpu
