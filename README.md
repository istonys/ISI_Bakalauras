# ISI Bakalauras: kalbos triukšmo šalinimas CNN modeliais

Ši saugykla skirta baigiamojo bakalauro darbo eksperimentams su vienkanaliu kalbos triukšmo šalinimu. Kode įgyvendinti CNN tipo modeliai, duomenų paruošimo skriptai, mokymo ir vertinimo eiga bei eksperimentų konfigūracijos.

Naudoti pagrindiniai duomenų rinkiniai:

- VoiceBank + DEMAND anglų kalbos eksperimentams;
- LIEPA-1 + DEMAND lietuvių kalbos eksperimentams.

## Struktūra

```text
src/        programinis kodas
configs/    eksperimentų konfigūracijos
scripts/    SLURM paleidimo skriptai HPC aplinkai
all_runs.csv galutinė eksperimentų rezultatų suvestinė
```

Sugeneruoti duomenys, modelių būsenos ir pilni rezultatų katalogai į saugyklą nekeliami.

## Aplinkos paruošimas

```bash
cd /scratch/lustre/home/$USER/bakalauras_audio_hpc
mkdir -p logs results checkpoints
bash scripts/setup_env.sh
source .venv/bin/activate
```

## Duomenų paruošimas

VoiceBank + DEMAND manifestai:

```bash
python -m src.prepare_voicebank_manifest \
  --data_root /scratch/lustre/home/$USER/data/VoiceBank_DEMAND \
  --output_dir /scratch/lustre/home/$USER/data/voicebank_28spk_manifest
```

LIEPA-1 + DEMAND rinkinys:

```bash
bash scripts/run_liepa_prep.sh
```

## Eksperimentų paleidimas

GPU patikrinimas:

```bash
sbatch scripts/test_gpu.sh
```

Sausas testas:

```bash
sbatch scripts/run_dry_run.sh
```

Modelių atranka:

```bash
sbatch scripts/run_screening_liepa.sh
```

Parametrų paieška:

```bash
sbatch scripts/run_parameter_sweep.sh
```

Pagrindiniai eksperimentai:

```bash
sbatch scripts/run_full_voicebank.sh
sbatch scripts/run_full_liepa_matched.sh
sbatch scripts/run_full_liepa_full.sh
```

Tarpkalbinis vertinimas:

```bash
sbatch scripts/run_cross_language.sh
```

## Rezultatų failai

Pagrindinė rezultatų suvestinė pateikiama `all_runs.csv`. Pilno vykdymo metu kiekvienam eksperimentui papildomai generuojami `history.csv`, `run_summary.json`, `file_metrics.csv`, mokymo kreivės, spektrogramų ir bangos formų paveikslai.

## Pastaba dėl duomenų ir modelių

Garso failai, LIEPA-1, DEMAND, VoiceBank + DEMAND duomenys, sugeneruoti triukšmingi įrašai, `results/`, `checkpoints/` ir kiti dideli failai į GitHub saugyklą nekeliami.
