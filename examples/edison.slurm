#! /bin/bash

#SBATCH -n 32
set -x
module load python/2.7-anaconda

source /project/projectdirs/m779/nbodykit/activate.sh

srun -n 16 python-mpi $NBKITBIN FOF <<EOF
nmin: 10
datasource:
    plugin: FastPM
    path: data/fastpm_1.0000
linklength: 0.2 
output: output/fof_ll0.200_1.0000.hdf5
calculate_initial_position: True
EOF

srun -n 16 python-mpi $NBKITBIN FFTPower <<EOF

mode: 2d      # 1d or 2d
Nmesh: 256     # size of mesh

# here are the input fields 
field:
  DataSource:
    plugin: FOFGroups
    path: output/fof_ll0.200_1.0000.hdf5
    m0: 2.27e12  # mass of a particle: use OmegaM * 27.75e10 * BoxSize ** 3/ Ntot
    rsd: z       # direction of RSD, usually use 'z', the default LOS direction.
#    select: Rank < 1000 # Limits to the first 1000 halos for abundance matching or
    select: LogMass > 13 # limit to the halos with logMass > 13 (LRGs). 
output: output/power_2d_fofgroups.dat  # output
EOF
