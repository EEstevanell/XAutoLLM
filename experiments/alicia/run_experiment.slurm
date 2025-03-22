#!/bin/bash
#SBATCH --job-name=autogoal-alicia    # Name of the job
#SBATCH --output=%j_alicia.out        # Name of the output file (using job ID)
#SBATCH --error=%j_alicia.err         # Name of the error file (using job ID)
#SBATCH --cpus-per-task=8             # Number of CPUs (matches the thread settings in main.py)
#SBATCH --mem=16G                     # Memory allocation (adjust based on your experiment needs)
#SBATCH --partition=postiguet1        # Partition name (adjusted to match your p.slurm example)
#SBATCH --gres=gpu:1                  # Request 1 GPU
#SBATCH --gres=shard:10               # Added shard allocation based on your p.slurm example
#SBATCH --time=72:00:00               # Maximum runtime of 72 hours (adjust as needed)

# Print job information
echo "Iniciando trabajo en `hostname` a las `date` $SLURM_IDX"
nvidia-smi

# Set current working directory
R=`pwd`
echo "Working directory: $R"

# Ensure our wrapper script is executable
chmod +x $R/experiments/alicia/run_experiment.sh

# Build the Docker image from the core Dockerfile
echo "Building Docker image from core Dockerfile..."
docker build --no-cache \
  --build-arg USER_ID=$(id -u) \
  --build-arg GROUP_ID=$(id -g) \
  -t autogoal-slurm-experiment \
  -f $R/dockerfiles/core/dockerfile $R

# Set environment variables for GPU and experiment config
echo "Running experiment container with GPU access..."
docker run \
  --name autogoal_${SLURM_JOB_ID} \
  --shm-size=8g \
  --gpus device=$SLURM_IDX \
  --rm \
  -v $R:/home/coder/autogoal \
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