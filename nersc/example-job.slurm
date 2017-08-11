#!/bin/bash
#SBATCH -p debug
#SBATCH -o nbkit-example
#SBATCH -n 16

# load nbodykit
source /usr/common/contrib/bccp/conda-activate.sh 3.6

# run the main nbodykit example
srun -n 16 python example.py
