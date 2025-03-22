#!/bin/bash
#SBATCH --job-name=mi_tarea          # Nombre del trabajo
#SBATCH --output=%j.out         # Nombre del archivo de salida
#SBATCH --error=%j.err          # Nombre del archivo de error
#SBATCH --cpus-per-task=1             # Número de CPUs por tarea
#SBATCH --mem=1G                      # Memoria por nodo
#SBATCH --partition=postiguet1           # Cola (partición) a la que enviar el trabajo
#SBATCH --gres=gpu:1
#SBATCH --gres=shard:10


# Aquí empieza la sección de comandos que se van a ejecutar

echo "Iniciando trabajo en `hostname` a las `date` $SLURM_IDX"
nvidia-smi

R=`pwd`
# Ejecuta tu programa aquí (reemplaza esto con lo que desees ejecutar)
# Ejecutar el contenedor con acceso a la GPU
docker build --no-cache --build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) -t pytorch-train3 .
#docker run --name $SLURM_JOB_ID --shm-size=1g --gpus device=$SLURM_IDX --rm -v $R:/workspace -e SCRIPT=/workspace/experiments/kan_heads/Code/polynomials/experiment_seeds_1.py pytorch-train python3 /workspace/experiments/kan_heads/Code/polynomials/experiment_seeds_1.py
#docker run --name $SLURM_JOB_ID --shm-size=1g --gpus device=$SLURM_IDX --rm -v $R:/workspace  pytorch-train3 
docker run --name $SLURM_JOB_ID --shm-size=1g --gpus device=$SLURM_IDX --rm -v /home/gplsi/fabio:/workspace  pytorch-train3
docker rmi pytorch-train3
echo "Trabajo finalizado a las `date`"
