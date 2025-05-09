#!/bin/bash
#SBATCH --job-name=autogoal-textgen
#SBATCH --partition=titan
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16 # This will be passed to the container
#SBATCH --mem=64G
#SBATCH --output=/home/coder/autogoal/experiments/text_generation/logs/slurm-%j.out
#SBATCH --error=/home/coder/autogoal/experiments/text_generation/logs/slurm-%j.err
#SBATCH --time=48:00:00

# --- Configuration ---
HOST_PROJECT_ROOT="/home/gplsi/estevanell/XAutoLLM" # Actual path on the Slurm node
CONTAINER_PROJECT_ROOT="/home/coder/autogoal"   # Desired path inside the container
DOCKER_IMAGE_NAME="autogoal-textgen-experiment"
# Dockerfile path relative to the HOST_PROJECT_ROOT
DOCKERFILE_PATH="$HOST_PROJECT_ROOT/dockerfiles/core/dockerfile"

# --- Output Directories ---
# Logs will be stored relative to the HOST_PROJECT_ROOT on the Slurm node
LOG_DIR="$HOST_PROJECT_ROOT/experiments/text_generation/logs"
mkdir -p "$LOG_DIR" # Ensure log directory exists

# --- Print Slurm and environment info ---
echo "===== Slurm Job Information ====="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_NODELIST: $SLURM_NODELIST"
echo "SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "CUDA_VISIBLE_DEVICES (original Slurm): $CUDA_VISIBLE_DEVICES" 
echo "Running on host: $(hostname)"
echo "Job started at: $(date)"
echo "==============================="

# Load modules if needed (uncomment and adjust for your cluster)
# module load cuda/11.3 # Example
# module load docker # Example, if docker is a module

# Activate your environment if needed (e.g., for Docker client if not in PATH)
# source /path/to/some_env/bin/activate

# --- Docker Image Management ---
echo "Checking for Docker image: $DOCKER_IMAGE_NAME..."
if docker image inspect "$DOCKER_IMAGE_NAME" &> /dev/null; then
  echo "Using existing Docker image '$DOCKER_IMAGE_NAME'."
else
  echo "Docker image '$DOCKER_IMAGE_NAME' not found. Building..."
  if [ -f "$DOCKERFILE_PATH" ]; then
    docker build --no-cache \
      --build-arg USER_ID=$(id -u) \
      --build-arg GROUP_ID=$(id -g) \
      -t "$DOCKER_IMAGE_NAME" \
      -f "$DOCKERFILE_PATH" "$HOST_PROJECT_ROOT" # Context is HOST_PROJECT_ROOT
    echo "Docker image '$DOCKER_IMAGE_NAME' built successfully."
  else
    echo "Error: Dockerfile not found at $DOCKERFILE_PATH. Cannot build image."
    exit 1
  fi
fi

# --- Run Experiment in Docker Container ---
echo "Running experiment in Docker container: $DOCKER_IMAGE_NAME..."
# The CUDA_VISIBLE_DEVICES for inside the container will be determined by --gpus all
# and what Slurm has allocated. Docker handles the mapping.
docker run \
  --name "autogoal_textgen_${SLURM_JOB_ID}" \
  --gpus all \
  --rm \
  -v "$HOST_PROJECT_ROOT:$CONTAINER_PROJECT_ROOT" \
  -e "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES" \
  -e "SLURM_JOB_ID=$SLURM_JOB_ID" \
  -e "SLURM_NODELIST=$SLURM_NODELIST" \
  -e "CPUS_PER_TASK=$SLURM_CPUS_PER_TASK" \
  -w "$CONTAINER_PROJECT_ROOT" \
  "$DOCKER_IMAGE_NAME" \
  bash "$CONTAINER_PROJECT_ROOT/experiments/text_generation/run_experiment.sh"

EXIT_CODE=$?
echo "Docker container finished with exit code: $EXIT_CODE"
echo "Job finished at: $(date)"
exit $EXIT_CODE
