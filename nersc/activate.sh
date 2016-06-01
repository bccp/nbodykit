#!/bin/bash

if [[ -n $BASH_VERSION ]]; then
    _SCRIPT_LOCATION=${BASH_SOURCE[0]}
elif [[ -n $ZSH_VERSION ]]; then
    _SCRIPT_LOCATION=${funcstack[1]}
else
    echo "Only bash and zsh are supported"
    return 1
fi

NBKITROOT=`dirname ${_SCRIPT_LOCATION}`
NBKITROOT=`readlink -f $NBKITROOT/..`

module load python/2.7-anaconda

source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

bcast ${NBKITROOT}/nersc/${NERSC_HOST}/nbodykit-dep.tar.gz ${NBKITROOT}/nersc/${NERSC_HOST}/nbodykit.tar.gz

function srun-nbkit {
    local np=
    if [ "x$1" == "x-n" ]; then
        shift
        np="-n $1"
        shift
    fi
    srun $np python-mpi /dev/shm/local/bin/nbkit.py $*
}

