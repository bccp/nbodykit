#! /bin/bash

#SBATCH -n 32
set -x
module load python/2.7-anaconda

source /project/projectdirs/m779/python-mpi/nersc/activate.sh
bcast -v ../../nbodykit-dep-${NERSC_HOST}.tar.gz ../../nbodykit.tar.gz

srun -n 16 python-mpi /dev/shm/local/bin/nbkit.py FFTCorrelation test_plaintext.params || exit
