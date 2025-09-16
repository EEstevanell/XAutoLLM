# XAutoLM: Meta-Learning Enhanced AutoML for Resource-Efficient Language Model Fine-Tuning

This repository contains the implementation of our paper submitted to ACL 2025.

> **Note**: This is an anonymous repository for peer review purposes, available at:  
> https://anonymous.4open.science/r/XAutoLLM-A010

XAutoLLM is built upon [AutoGOAL](https://github.com/autogoal/autogoal), an automatic machine learning toolkit. We extend AutoGOAL's capabilities with meta-learning mechanisms specifically designed for Language Model fine-tuning optimization.

## Abstract

Experts in machine learning distinguish themselves from amateurs by leveraging domain knowledge to effectively navigate the myriad decisions involved in model selection and hyperparameter optimisation. This distinction is especially critical for Language Models (LMs), whose repeated fine-tuning trials incur substantial computational overhead and raise environmental concerns. Yet, no single AutoML framework simultaneously addresses both model selection and hyperparameter optimisation for resource-efficient LM fine-tuning.

We propose XAutoLLM, which integrates meta-learning to warm start the search space. By drawing on task- and system-level meta-features, XAutoLLM reuses insights from previously tuned LMs on related tasks, enabling resource-friendly, Green AI fine-tuning that balances state-of-the-art outcomes with minimised computational overhead.

## Requirements

### Hardware Requirements

- CUDA-capable GPUs for transformer-based models
- Minimum 16GB GPU memory recommended
- 32GB system RAM recommended

### Software Requirements

- Docker with GPU support (recommended)
- Python 3.8+
- CUDA Toolkit 11.7+ (for GPU support)

## Reproducing Experimental Results

### Docker Setup (Recommended)

1. **Build the Docker image:**
```bash
make docker
```

2. **Run the container with GPU support:**
```bash
make container-gpu
```

### Running Experiments

For detailed experimental setup and configuration options, please refer to [experiments/Readme.md](experiments/Readme.md).

1. **Enter the experiments directory:**
```bash
cd experiments
```

2. **Run all experiments and analysis:**
```bash
make paper
```

This will:
1. Execute multi-objective experiments
2. Execute single-objective experiments
3. Analyze results using appropriate modules
4. Generate figures and tables

For more granular control:
- `make run-multi-objective` - Run only multi-objective experiments
- `make run-single-objective` - Run only single-objective experiments
- `make analyze-results` - Analyze existing results

## Project Structure

```
XAutoLLM/
├── autogoal/           # Core AutoML framework (forked from AutoGOAL)
├── experiments/        # Experimental framework and results
│   ├── configs/       # Experiment configurations
│   ├── data/         # Datasets and experience store
│   ├── output/       # Results and logs
│   └── src/          # Source code for experiments
└── dockerfiles/       # Docker configuration files
```

## Acknowledgments

This work builds upon [AutoGOAL](https://github.com/autogoal/autogoal). We thank the original authors for their foundational work in automated machine learning.


