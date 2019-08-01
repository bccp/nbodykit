#!/bin/bash
#SBATCH -p debug
#SBATCH -o nbkit-example
#SBATCH -n 16

# load nbodykit
source /global/common/software/m3035/conda-activate.sh 3.7

# run the main nbodykit example
srun -n 16 python example.py
