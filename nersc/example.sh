#!/bin/bash
#SBATCH -p debug
#SBATCH -o nbkit-example
#SBATCH -n 16

# You can also allocate the nodes with salloc
#
# salloc -n 16
#
# and type the commands in the shell obtained from salloc

module unload python
module load python/3.5-anaconda

source /usr/common/contrib/bccp/nbodykit/activate.sh dev

# regular nbodykit command lines
# replace nbkit.py with srun-nbkit

srun -n 16 python-mpi -c 'from nbodykit.lab import *;print(FFTPower)'

# You can also do this in an interactive shell
# e.g.

