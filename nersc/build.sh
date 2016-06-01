#!/bin/bash

source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh

mkdir -p ${NERSC_HOST}
MPICC=cc bundle-pip ${NERSC_HOST}/nbodykit-dep.tar.gz -r ../requirements.txt 
bundle-pip ${NERSC_HOST}/nbodykit.tar.gz ..
