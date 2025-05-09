#!/bin/bash
# run_experiment.sh - Executes the experiment script inside the Docker container.

# Fail on error
set -e

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
echo "PYTHONPATH (updated): $PYTHONPATH"

# Set threading environment variables based on CPUS_PER_TASK from Slurm
# Default to 1 if CPUS_PER_TASK is not set or empty
EFFECTIVE_CPUS=${CPUS_PER_TASK:-1}
export OMP_NUM_THREADS=$EFFECTIVE_CPUS
export MKL_NUM_THREADS=$EFFECTIVE_CPUS
export NUMEXPR_NUM_THREADS=$EFFECTIVE_CPUS
export OPENBLAS_NUM_THREADS=$EFFECTIVE_CPUS
export VECLIB_MAXIMUM_THREADS=$EFFECTIVE_CPUS
export BLIS_NUM_THREADS=$EFFECTIVE_CPUS

echo "--- Threading Environment Variables ---"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS: $OPENBLAS_NUM_THREADS"
echo "VECLIB_MAXIMUM_THREADS: $VECLIB_MAXIMUM_THREADS"
echo "BLIS_NUM_THREADS: $BLIS_NUM_THREADS"
echo "-------------------------------------"

# Change to the project root directory. This is good practice.
cd "$PROJECT_ROOT_IN_CONTAINER"
echo "Changed directory to: $(pwd)"

echo "Executing Python experiment script: $EXPERIMENT_PYTHON_SCRIPT"
python3 "$EXPERIMENT_PYTHON_SCRIPT" "$@"

EXIT_CODE=$?
echo "Python script finished with exit code: $EXIT_CODE"
echo "===== Exiting Docker Container: run_experiment.sh ====="
exit $EXIT_CODE
