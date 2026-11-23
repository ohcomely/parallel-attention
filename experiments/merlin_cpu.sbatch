#!/bin/bash
# One allocation of a Merlin7 CPU experiment: 8 EPYC 7742 nodes x 16 ranks x 8 cores (P up to 128).
#     sbatch experiments/merlin_cpu.sbatch SPEC [JOB]      (JOB = which of the spec's `jobs` to run; default all)
# Submit from the repository root after building (cmake -S . -B build && cmake --build build -j).
#SBATCH --cluster=merlin7
#SBATCH --partition=hourly
#SBATCH --time=01:00:00
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=8
#SBATCH --hint=nomultithread
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --job-name=attn
#SBATCH --output=attn-%j.out
set -euo pipefail

module purge
module load gcc/14.3.0 openmpi/5.0.8   # plus an OpenBLAS module (find it with `module search openblas`)
# Optional, recommended by the Merlin7 docs for inter-node Open MPI traffic (preview provider):
# export FI_PROVIDER=lnx FI_LNX_PROV_LINKS="shm+cxi:cxi0"

spec=${1:?usage: sbatch experiments/merlin_cpu.sbatch SPEC [JOB]}
job=${2:+--job $2}
python3 experiments/run.py "$spec" $job
