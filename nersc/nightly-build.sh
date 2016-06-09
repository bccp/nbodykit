#!/bin/bash -l

# change to a temporary directory
cd $(mktemp -d)

# make the build directory
mkdir -p ${NERSC_HOST}

# activate python-mpi-bcast
source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh

# build the "latest" source from the "develop" branch
bundle-pip ${NERSC_HOST}/nbodykit-latest.tar.gz git+https://github.com/bccp/nbodykit.git@develop
rsync ${NERSC_HOST}/nbodykit-latest.tar.gz /usr/common/contrib/bccp/nbodykit/

# build the "stable" source from the "master" branch
bundle-pip ${NERSC_HOST}/nbodykit-stable.tar.gz git+https://github.com/bccp/nbodykit.git@master
rsync ${NERSC_HOST}/nbodykit-stable.tar.gz /usr/common/contrib/bccp/nbodykit/

# build the dependencies from the requirements.txt file
MPICC=cc bundle-pip ${NERSC_HOST}/nbodykit-dep.tar.gz -r https://raw.githubusercontent.com/bccp/nbodykit/develop/requirements.txt
rsync ${NERSC_HOST}/nbodykit-dep.tar.gz /usr/common/contrib/bccp/nbodykit/

# remove the temporary directory
rm -r $(pwd)

