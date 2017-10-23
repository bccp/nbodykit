#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -J run.$1
#SBATCH -o emu-bench.$1
#SBATCH -n $1
#SBATCH -t 10:00

source /usr/common/contrib/bccp/conda-activate.sh 3.6
time bcast-pip nbodykit==0.2.6
time bcast-pip https://github.com/rainwoodman/fastpm-python/archive/master.zip

echo ===== Running with $1 cores =====
srun -n $1 python -u emulator.py
EOF
