#!/bin/bash
set -e

# -----------------------------------------------------------------------------
# MIF HPC: kai kurie mazgai reikalauja `module load` prieš naudojant python /
# cuda. Atblokuokite reikalingas eilutes patikrinę `module avail` aktyviame
# mazge. Jei mazgas pats turi Python 3.10+ ir CUDA driverius, šias eilutes
# galima palikti užkomentuotas.
# -----------------------------------------------------------------------------
# module purge
# module load python/3.10
# module load cuda/12.1
# module load gcc

# Sukuriame ir aktyvuojame venv lokaliame projekto kataloge.
python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

# V100 GPU paprastai veikia su CUDA palaikančiu PyTorch build'u. Jei šis
# variantas neveiktų dėl HPC driverių, pakeisk index URL pagal faktinę
# CUDA versiją (pvz. cu118, cu124 ir t.t.).
python -m pip install torch --index-url https://download.pytorch.org/whl/cu121

python -m pip install -r requirements.txt

python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
