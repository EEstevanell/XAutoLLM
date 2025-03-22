#!/bin/bash
# Wrapper script for running the experiment in a Slurm environment

# Export necessary environment variables for PyTorch and CUDA
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export VECLIB_MAXIMUM_THREADS=8
export BLIS_NUM_THREADS=8

# Set CUDA visible devices to match Slurm's allocation
if [ -n "$SLURM_IDX" ]; then
  export CUDA_VISIBLE_DEVICES=$SLURM_IDX
else
  # Default to device 0 if not set by Slurm
  export CUDA_VISIBLE_DEVICES=0
fi

# Print environment information
echo "===== Environment Information ====="
echo "Running on node: $(hostname)"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "Current directory: $(pwd)"
echo "=================================="

# Run the main experiment script
cd /home/coder/autogoal
python /home/coder/autogoal/experiments/alicia/src/main.py "$@"