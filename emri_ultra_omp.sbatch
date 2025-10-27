#!/bin/bash
#SBATCH --job-name=emri_ultra
#SBATCH --account=nbody_int
#SBATCH --partition=interactive
#SBATCH --qos=djroomba_int
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem=128G
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

module --force purge
module load StdEnvACCRE/2023
module load openmpi/4.1.5
module load mpi4py/4.0.3

# --- Activate your working venv (where you built everything from source)
source /panfs/accrepfs.vampire/home/bustam1/emri_tdi_sampler/.venv/bin/activate

# --- Results directory per job
RESULTS="ultra_results_mpi_${SLURM_JOB_ID}"
rm -rf "$RESULTS"

# --- Run
mpiexec -n "${SLURM_NTASKS}" \
  python main_ultranest.py \
    --use-mpi \
    --prior zoom \
    --samples m1 m2 p0 e0 \
    --tobs 0.1 \
    --results "${RESULTS}"
