#!/bin/bash

#SBATCH -p {{ partition }}
#SBATCH -J {{ job }}.{{ cores }}
#SBATCH -o {{ output_file }}
#SBATCH -N {{ nodes }}
#SBATCH -t {{ time }}
{{ haswell_config }}

# activate environment
source /usr/common/contrib/bccp/conda-activate.sh {{ python_version }}

# clone nbodykit
git clone https://github.com/bccp/nbodykit

# checkout correct source version
cd nbodykit
git checkout {{ tag }}

# install correct nbodykit version to computing nodes
bcast-pip .

echo ===== Running with {{ cores }} cores =====
srun -n {{ cores }} python -u run-tests.py {{ benchname }} -m {{ sample }} --bench --no-build --bench-dir {{ benchdir }}
