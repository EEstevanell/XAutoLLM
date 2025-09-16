# XAutoLM: Meta-Learning Enhanced AutoML for Resource-Efficient Language Model Fine-Tuning

XAutoLM is the official public release of the system described in the paper "XAutoLM: Efficient Fine-Tuning of Language Models via Meta-Learning and AutoML," accepted for presentation at EMNLP 2025. The toolkit extends [AutoGOAL](https://github.com/autogoal/autogoal) with meta-learning warm starts aimed at reducing the search cost of language model fine-tuning across both text-classification and text-generation tasks. The repository ships the full experimental pipeline, pretrained experience stores, and the datasets used in the publication.

**Preprint (arXiv):** <https://arxiv.org/abs/2508.00924>  

## Highlights

- Meta-learning warm starts reuse task and system meta-features to shrink the exploration space for new tasks.
- Unified AutoML interface for single-objective and multi-objective optimisation with GPU-aware scheduling.
- Turn-key Docker and Python tooling for reproducing the paper results end to end.
- Bundled datasets and experience stores under `experiments/text_classification/data` and `experiments/text_generation/data` enable offline analysis.

## Getting Started

### Docker (recommended)

1. Build the GPU-enabled image:
   ```bash
   make docker
   ```
2. Launch the container with GPU access:
   ```bash
   make container-gpu
   ```
   The container mounts the project at `/home/coder/autogoal`, matching the paths referenced in the experiment scripts.

### Local installation

```bash
python -m venv .venv
source .venv/bin/activate   # On Windows use: .venv\Scripts\activate
pip install --upgrade pip
pip install -e .
pip install -e .[contrib]
pip install -r experiments/requirements.txt
```

For local execution you will need at least one CUDA-capable GPU (16 GB VRAM recommended) and 32 GB of host RAM for the largest benchmarks.

## Reproducing the Experiments

The experiments are organised into two suites: `text_classification` (binary and multi-class problems) and `text_generation` (summarisation and question answering). Each suite exposes the same command-line interface.

### Environment preparation

From the repository root, add the `experiments` directory to `PYTHONPATH` so the experiment modules resolve correctly:

```bash
export PYTHONPATH="$(pwd)/experiments:${PYTHONPATH}"      # Linux/macOS
set PYTHONPATH=%cd%\experiments;%PYTHONPATH%              # Windows PowerShell
```

### Text classification suite

```bash
cd experiments/text_classification
python src/execute_experiments.py --experiment_type multi
python src/execute_experiments.py --experiment_type single
```

- Results are stored under `experiments/text_classification/output`.
- Pre-collected experience for analysis lives in `experiments/text_classification/data/experience_store`.
- `run_experiment.sh` can be invoked inside the Docker container or via SLURM (`p4.slurm`) to orchestrate distributed runs.

### Text generation suite

```bash
cd experiments/text_generation
python src/execute_experiments.py --experiment_type multi
python src/execute_experiments.py --experiment_type single
```

- Outputs are written to `experiments/text_generation/output`.
- Warm-start experience data used in the paper is provided in `experiments/text_generation/data`.
- For containerised or HPC execution, use `run_experiment.sh` (paired with `p3.slurm`).

### Analysis and experience syncing

Each suite offers dedicated analysis utilities under `src/analysis/`. After completing new runs you can update the bundled experience stores to include fresh results:

```bash
python src/analysis/multi_objective/main.py
python src/analysis/single_objective/main.py
```

Copying experience into the analysis store ensures the reporting scripts pick up your executions:

```bash
rsync -av ~/.autogoal/data/experience_store/ experiments/text_classification/data/experience_store/
rsync -av ~/.autogoal/data/experience_store/ experiments/text_generation/data/experience_store/
```

## Data availability

All datasets referenced in the publication are distributed with this repository. Raw and processed assets can be found at:

- `experiments/text_classification/data`
- `experiments/text_generation/data`

Refer to the README files inside each dataset subdirectory for licensing notes and provenance.

## Repository layout

```
XAutoLM/
|- autogoal/                 # Core AutoGOAL fork with XAutoLM extensions
|- autogoal-contrib/         # Optional integrations packaged with the project
|- autogoal-remote/          # Remote execution helpers
|- docs/                     # Paper drafts and supplementary material
|- dockerfiles/              # Container definitions used for reproduction
|- experiments/              # Full experimental pipeline (see above)
|  |- text_classification/   # Classification-specific configs, data, code, outputs
|  |- text_generation/       # Generation-specific configs, data, code, outputs
|  |- makefile               # Legacy aggregate Make targets (container paths)
|  |- requirements.txt       # Experiment dependencies
|- plots/                    # Figure outputs generated by analysis scripts
|- scripts/                  # Utility scripts for automation and maintenance
|- CITATION.cff              # Citation metadata (AutoGOAL foundation)
|- LICENSE                   # Project licence
|- Readme.md                 # This document
```

## Citation

If you build on XAutoLM, please cite our EMNLP paper. We directly acknowledge and cite AutoGOAL throughout the project; its reference is reproduced below for completeness should you also wish to acknowledge the foundation.

**XAutoLM (EMNLP 2025, arXiv preprint).** The EMNLP proceedings citation will be added once available; in the interim please reference the arXiv version:

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

**AutoGOAL foundation (for reference).**

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

## Acknowledgements

We thank the AutoGOAL community for the foundational infrastructure and the reviewers whose feedback helped shape the EMNLP 2025 release. Please open an issue if you spot documentation gaps or have questions about running additional benchmarks.
