#!/bin/bash
#SBATCH --job-name=ibermat    # Name of the job
#SBATCH --output=%j_ibermat.out        # Name of the output file (using job ID)
#SBATCH --error=%j_ibermat.err         # Name of the error file (using job ID)
#SBATCH --cpus-per-task=16             # Number of CPUs (matches the thread settings in main.py)
#SBATCH --mem=24GB                     # Memory allocation (adjust based on your experiment needs)
#SBATCH --partition=titan        # Partition name (adjusted to match your p.slurm example)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --gres=shard:10               # Added shard allocation based on your p.slurm example
#SBATCH --time=48:00:00               # Maximum runtime of 72 hours (adjust as needed)

# Print job information
echo "Iniciando trabajo en `hostname` a las `date` $SLURM_IDX"
nvidia-smi

# Set current working directory
R=`pwd`
echo "Working directory: $R"

# Set the correct XAutoLLM directory
XAUTOLLM_DIR="/home/gplsi/estevanell/XAutoLLM"
echo "XAutoLLM directory: $XAUTOLLM_DIR"

# Ensure our wrapper script is executable
chmod +x $R/run_experiment.sh

# Check if the Docker image already exists
if docker image inspect autogoal-slurm-experiment &> /dev/null; then
  echo "Using existing Docker image 'autogoal-slurm-experiment'..."
else
  # Build the Docker image from the core Dockerfile
  echo "Building Docker image from core Dockerfile..."
  docker build --no-cache \
    --build-arg USER_ID=$(id -u) \
    --build-arg GROUP_ID=$(id -g) \
    --network=host \
    -t autogoal-slurm-experiment \
    -f $XAUTOLLM_DIR/dockerfiles/core/dockerfile $XAUTOLLM_DIR
fi

# Set environment variables for GPU and experiment config
echo "Running experiment container with GPU access..."
docker run \
  --name autogoal_${SLURM_JOB_ID} \
  --gpus device=$SLURM_IDX \
  --rm \
  --network=host \
  -v $XAUTOLLM_DIR:/home/coder/autogoal \
  -e PYTHONPATH=/home/coder/autogoal \
  -e EXPERIMENT_ID=alicia_${SLURM_JOB_ID} \
  -e SLURM_JOB_ID=$SLURM_JOB_ID \
  -e SLURM_IDX=$SLURM_IDX \
  -e SLURM_NODELIST=$SLURM_NODELIST \
  autogoal-slurm-experiment \
  /home/coder/autogoal/experiments/alicia/run_experiment.sh

# Clean up
echo "Removing Docker image..."
docker rmi autogoal-slurm-experiment

echo "Trabajo finalizado a las `date`"