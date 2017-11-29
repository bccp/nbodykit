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
trap "rm -rf $scratch" EXIT

# install correct nbodykit version to computing nodes
bcast-pip git+git://github.com/bccp/nbodykit.git@{{ tag }}
bcast-pip git+git://github.com/bccp/runtests.git

# clone nbodykit (so we can run benchmarks)
git clone https://github.com/bccp/nbodykit
cd nbodykit
git checkout benchmark-tests

# call run-tests with desired number of cores
echo ===== Running with {{ cores }} cores =====
python -u run-tests.py {{ benchname }} --mpirun "srun -n {{ cores }}" {{ sample }} --bench --no-build --bench-dir {{ benchdir }} -s
