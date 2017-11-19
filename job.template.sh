#!/bin/bash

#SBATCH -p {{ partition }}
#SBATCH -J {{ job }}.{{ cores }}
#SBATCH -o {{ output_file }}
#SBATCH -N {{ nodes }}
#SBATCH -t {{ time }}
{{ haswell_config }}

# activate environment
source /usr/common/contrib/bccp/conda-activate.sh {{ python_version }}

# make a temp directory and cd there
scratch=$(mktemp -d)
cd $scratch;

# remove tmp directory on EXIT
#trap "rm -rf $scratch" EXIT

# clone nbodykit
git clone https://github.com/bccp/nbodykit

# checkout correct source version
cd nbodykit
git checkout {{ tag }}

# install correct nbodykit version to computing nodes
bcast-pip git+git://github.com/nickhand/nbodykit.git@correlation
bcast-pip git+git://github.com/bccp/runtests.git

# checkout the benchmarks
git checkout benchmark-tests

echo ===== Running with {{ cores }} cores =====
python -u run-tests.py {{ benchname }} --mpirun "srun -n {{ cores }}" -m {{ sample }} --bench --no-build --bench-dir {{ benchdir }}
