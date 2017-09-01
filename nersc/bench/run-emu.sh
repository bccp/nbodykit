#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -J run.$1
#SBATCH -o emu-bench.$1
#SBATCH -n $1
#SBATCH -t 30:00

source /usr/common/contrib/bccp/conda-activate.sh 3.6
bcast-pip https://github.com/rainwoodman/fastpm-python/archive/master.zip

echo ===== Running with $1 cores =====
srun -n $1 python -u emulator.py
EOF
