#!/bin/bash
#SBATCH --job-name=nc_trento
#SBATCH --output=nc_%j.out
#SBATCH --error=nc_%j.err

#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G

# Load modules (adjust to your cluster)
module load gcc
module load cmake
module load python

# Activate your environment if needed
# source ~/env/bin/activate

# Set TRENTO path (IMPORTANT)
export TRENTO_BIN=$HOME/trento/build/src/trento

# Go to working directory
cd $SLURM_SUBMIT_DIR

echo "Running on $SLURM_NODELIST"
echo "TRENTO = $TRENTO_BIN"

python nc_convergence_hpc.py
