#!/bin/bash

sbatch <<EOF
#!/bin/bash
#SBATCH -C haswell
#SBATCH -p debug
#SBATCH -J run.$1
#SBATCH -o convpower-bench.$1
#SBATCH -n $1
#SBATCH -t 30:00

source /usr/common/contrib/bccp/conda-activate.sh 3.6

echo ===== Running with $1 cores =====
srun -n $1 python -u convpower.py North 1
EOF
