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

# --- BitsAndBytes Installation ---
echo "--- Checking and installing bitsandbytes ---"
if ! python3 -c "import bitsandbytes" > /dev/null 2>&1; then
    echo "bitsandbytes is not installed. Installing..."
    # Determine CUDA version
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $5}' | cut -d',' -f1)

    if [[ "$CUDA_VERSION" == "12.8"* || "$CUDA_VERSION" == "12.9"* ]]; then
        echo "CUDA 12.8 or 12.9 detected. Installing compatible bitsandbytes."
        pip install bitsandbytes
    else
        echo "CUDA version $CUDA_VERSION detected. Installing bitsandbytes (may require manual configuration)."
        pip install bitsandbytes
    fi
else
    echo "bitsandbytes is already installed."
fi
echo "---------------------------------------------"

echo "Executing Python experiment script: $EXPERIMENT_PYTHON_SCRIPT"
python3 "$EXPERIMENT_PYTHON_SCRIPT" "$@"

EXIT_CODE=$?
echo "Python script finished with exit code: $EXIT_CODE"
echo "===== Exiting Docker Container: run_experiment.sh ====="
exit $EXIT_CODE
