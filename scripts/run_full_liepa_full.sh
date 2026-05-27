#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH -t 24:00:00
#SBATCH -o logs/full_liepa_full_%j.out
#SBATCH -e logs/full_liepa_full_%j.err

set -e
cd /scratch/lustre/home/$USER/bakalauras_audio_hpc
mkdir -p logs

source .venv/bin/activate
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# ~67h LIEPA su 60 epochu gali pereiti 24h riba. Jei taip nutiks,
# paleisk i antra job'a su --resume ant ankstesnio checkpoint:
#   sbatch scripts/run_full_liepa_full.sh   # 1-as paleidimas
#   # ... jei nutruko, tesia nuo paskutinio checkpoint ...
#   sbatch scripts/run_full_liepa_full.sh   # 2-as paleidimas (--resume automatiskai)
# Arba rankiniu budu:
#   python -m src.train --config configs/full_liepa_full_selected.yaml \
#       --resume checkpoints/full_liepa_full_<model>_best.pth
python -m src.train --config configs/full_liepa_full_selected.yaml
