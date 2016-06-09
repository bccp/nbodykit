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
module load python/2.7-anaconda

source /usr/common/contrib/bccp/nbodykit/activate.sh

# regular nbodykit command lines
# replace nbkit.py with srun-nbkit

srun-nbkit -n 16 FFTPower --help

# You can also do this in an interactive shell
# e.g.

