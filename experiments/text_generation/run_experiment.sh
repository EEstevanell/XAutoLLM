#!/bin/bash
# run_experiment.sh - Executes the experiment script inside the Docker container.

PROJECT_ROOT_IN_CONTAINER="/home/coder/autogoal" # This path is how it's mounted in the container
EXPERIMENT_PYTHON_SCRIPT="$PROJECT_ROOT_IN_CONTAINER/experiments/text_generation/src/execute_experiments.py"

echo "===== Inside Docker Container: run_experiment.sh ====="
echo "Timestamp: $(date)"
echo "Running on container host: $(hostname)" # This will be the container's hostname
echo "Current directory: $(pwd)"
echo "User: $(whoami)"
echo "--- Environment Variables Passed from Slurm/Docker Run ---"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES" # Critical for GPU usage
echo "CPUS_PER_TASK: $CPUS_PER_TASK" # Passed from Slurm for threading
echo "PYTHONPATH (initial): $PYTHONPATH"
echo "----------------------------------------------------"

# Set Python Path within the container
export PYTHONPATH="$PROJECT_ROOT_IN_CONTAINER:${PYTHONPATH}"
export PYTHONPATH="/home/coder/autogoal/experiments:${PYTHONPATH}"
export PYTHONPATH="/home/coder/autogoal/experiments/text_classification:${PYTHONPATH}"
echo "PYTHONPATH (updated): $PYTHONPATH"

# Change to the project root directory. This is good practice.
cd "$PROJECT_ROOT_IN_CONTAINER"
echo "Changed directory to: $(pwd)"

echo "Executing Python experiment script: $EXPERIMENT_PYTHON_SCRIPT"
python3 "$EXPERIMENT_PYTHON_SCRIPT" "$@"

EXIT_CODE=$?
echo "Python script finished with exit code: $EXIT_CODE"
echo "===== Exiting Docker Container: run_experiment.sh ====="
exit $EXIT_CODE
