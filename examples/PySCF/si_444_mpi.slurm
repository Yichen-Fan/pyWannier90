#!/usr/bin/env bash
#SBATCH --job-name=si444-vc-dlwf
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=mei3
#SBATCH --time=01:00:00
#SBATCH --output=si-444-vc-%j.out
#SBATCH --error=si-444-vc-%j.err

set -e
set -o pipefail

source /home/yf144/anaconda3/etc/profile.d/conda.sh
conda activate pyscf211
source "$HOME/.local/wannier90-dlwf-4.0-mpi-conda/activate-wannier90-mpi.sh"

export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

PYTHON_SCRIPT="si_444_valence_conduction_v4.py"
SCF_THREADS="$SLURM_NTASKS"
MPI_RANKS="$SLURM_NTASKS"

echo "Python: $(command -v python)"
echo "MPI launcher: $(command -v mpiexec)"
echo "Wannier90 library: $PYWANNIER90_LIB"
echo "Allocated CPUs: $SLURM_NTASKS"

# Stage 1: use all four allocated CPUs as threads in one PySCF process.  The
# srun step gives that process all four cores instead of binding it to one.
export PYWANNIER90_STAGE=scf
export PYSCF_SCF_THREADS="$SCF_THREADS"
export OMP_NUM_THREADS="$SCF_THREADS"
export OPENBLAS_NUM_THREADS="$SCF_THREADS"
export MKL_NUM_THREADS="$SCF_THREADS"
export NUMEXPR_NUM_THREADS="$SCF_THREADS"

echo "Starting threaded PySCF stage with $SCF_THREADS threads"
srun --exclusive \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$SCF_THREADS" \
  --cpu-bind=cores \
  python -u "$PYTHON_SCRIPT"

# Stage 2: reuse the same four CPUs as four single-threaded MPI ranks.  Rank 0
# reads the saved orbitals and broadcasts them before entering Wannier90.
export PYWANNIER90_STAGE=localize
export PYWANNIER90_EXPECT_MPI_RANKS="$MPI_RANKS"
export PYSCF_SCF_THREADS=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Starting Wannier90 stage with $MPI_RANKS MPI ranks"
mpiexec --bind-to core -n "$MPI_RANKS" \
  python -u "$PYTHON_SCRIPT"
