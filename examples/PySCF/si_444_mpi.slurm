#!/usr/bin/env bash
#SBATCH --job-name=pyscf-w90-dlwf
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=mei3
#SBATCH --time=01:00:00
#SBATCH --output=si-444-%j.out
#SBATCH --error=si-444-%j.err

set -euo pipefail

source /home/yf144/anaconda3/etc/profile.d/conda.sh
conda activate pyscf211
source "$HOME/.local/wannier90-dlwf-4.0-mpi-conda/activate-wannier90-mpi.sh"

# Prevent each MPI process from creating an additional threaded BLAS team.
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export MKL_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$SLURM_CPUS_PER_TASK"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

echo "Python: $(command -v python)"
echo "MPI launcher: $(command -v mpiexec)"
echo "Wannier90 library: $PYWANNIER90_LIB"
echo "MPI ranks: $SLURM_NTASKS"

# Rank 0 performs the PySCF SCF and broadcasts its orbitals.  All ranks then
# enter W90, where the 64 k-points are distributed across the four ranks.
# Let Slurm capture stdout/stderr using the directives above.
mpiexec -n "$SLURM_NTASKS" python -u si_444.py
