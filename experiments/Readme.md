# XAutoLM Experiments

This directory aggregates the assets required to reproduce the experiments reported in the paper "XAutoLM: Efficient Fine-Tuning of Language Models via Meta-Learning and AutoML" (EMNLP 2025). The pipeline is split into two suites: text classification and text generation, sharing a common design for configuration management, execution, and analysis.

## Overview

```
experiments/
|- makefile                    # Container-oriented shortcuts (legacy)
|- requirements.txt            # Python dependencies for both suites
|- setup_pythonpath.sh         # Helper for container execution paths
|- text_classification/
|  |- configs/                 # YAML experiment definitions (single & multi objective)
|  |- data/                    # Datasets and warm-start experience stores
|  |- output/                  # Generated metrics, logs, tables, and plots
|  |- src/                     # Execution, analysis, and visualisation code
|  |- run_experiment.sh        # Docker / SLURM entrypoint
|  |- p4.slurm                 # SLURM script for multi-GPU clusters
|- text_generation/
   |- configs/
   |- data/
   |- output/
   |- src/
   |- run_experiment.sh
   |- p3.slurm
```

Both `data/` directories contain the corpora, metadata, and pre-collected experience referenced in the paper:

- `experiments/text_classification/data`
- `experiments/text_generation/data`

Refer to the `README` files within each dataset folder for provenance and licence details.

## Environment setup

Follow the installation steps from the repository root README or set up a virtual environment from here:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

If you executed `pip install -e .` from the repository root, the core XAutoLM packages will already be available. Ensure at least one CUDA-capable GPU (16 GB VRAM recommended) is visible before launching experiments.

Add the `experiments` folder to `PYTHONPATH` to resolve intra-package imports:

```bash
export PYTHONPATH="$(pwd):${PYTHONPATH}"                # Linux/macOS
set PYTHONPATH=%cd%;%PYTHONPATH%                        # Windows PowerShell
```

## Running the suites

### Text classification

```bash
cd text_classification
python src/execute_experiments.py --experiment_type multi
python src/execute_experiments.py --experiment_type single
```

- Logs and intermediate artefacts are stored in `text_classification/output`.
- Warm-start experience used by the analysis scripts resides in `text_classification/data/experience_store`.
- For containerised or SLURM-based execution reuse `run_experiment.sh` together with `p4.slurm`.

### Text generation

```bash
cd text_generation
python src/execute_experiments.py --experiment_type multi
python src/execute_experiments.py --experiment_type single
```

- Outputs mirror the classification suite and live under `text_generation/output`.
- Prebuilt experience snapshots are bundled in `text_generation/data/experience_store`.
- Use `run_experiment.sh` (optionally scheduled via `p3.slurm`) for automated GPU orchestration.

### Analysis utilities

Each suite exposes mirrored analysis entry points:

```bash
# From text_classification
python src/analysis/multi_objective/main.py
python src/analysis/single_objective/main.py

# From text_generation
python src/analysis/multi_objective/main.py
python src/analysis/single_objective/main.py
```

These scripts expect the relevant `data/experience_store` directory to hold the experience you wish to analyse. To include fresh executions, copy data from the runtime store created at `~/.autogoal/data/experience_store/` into the respective `data/experience_store/` folder.

## Data synchronisation

Synchronise runtime experiences into the repository snapshots to keep analyses reproducible:

```bash
rsync -av ~/.autogoal/data/experience_store/ text_classification/data/experience_store/
rsync -av ~/.autogoal/data/experience_store/ text_generation/data/experience_store/
```

(Replace `rsync` with `robocopy` on Windows or a manual copy if preferred.)

## Citation

If you build on XAutoLM, please cite our EMNLP paper. We already include the AutoGOAL reference within this repository, and we reproduce it below for completeness should you also wish to acknowledge it in derivative work.

- **XAutoLM (EMNLP 2025, arXiv preprint):**
  ```bibtex
  @misc{estevanellvalladares2025xautolmefficientfinetuninglanguage,
    title        = {XAutoLM: Efficient Fine-Tuning of Language Models via Meta-Learning and AutoML},
    author       = {Ernesto L. Estevanell-Valladares and Suilan Estevez-Velarde and Yoan Gutierrez and Andres Montoyo and Ruslan Mitkov},
    year         = {2025},
    eprint       = {2508.00924},
    archivePrefix = {arXiv},
    primaryClass = {cs.CL},
    url          = {https://arxiv.org/abs/2508.00924}
  }
  ```
  The EMNLP proceedings citation will replace the entry above once the anthology reference is published.

- **AutoGOAL foundation (reference copy):**
  ```bibtex
  @article{estevez-velarde2020autogoal,
    title   = {General-purpose hierarchical optimisation of machine learning pipelines with grammatical evolution},
    author  = {Suilan Estevez-Velarde and Yoan Gutierrez and Yudivian Almeida-Cruz and Andres Montoyo},
    journal = {Information Sciences},
    volume  = {511},
    pages   = {283--303},
    year    = {2020},
    doi     = {10.1016/j.ins.2020.07.035}
  }
  ```

## Support

Open an issue if any instructions are unclear or if additional metadata is required for your reproducibility package.
