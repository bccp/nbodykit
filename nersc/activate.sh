#!/bin/bash

# can supply 0 or 1 argument
if [ "$#" -gt 1 ]; then
    echo "usage: activate.sh latest|stable|v2.0"
    exit 1
fi

# if no version provided, use 'stable'
if [ $# -eq 0 ]; then
    version="stable"
elif [[ "$1" != "stable" && "$1" != "latest" && "$1" != "v2.0" ]]; then
    echo "valid version names: 'stable', 'latest', 'v2.0'"
    exit 1
else
    version=$1
fi

if [[ -n $BASH_VERSION ]]; then
    _SCRIPT_LOCATION=${BASH_SOURCE[0]}
elif [[ -n $ZSH_VERSION ]]; then
    _SCRIPT_LOCATION=${funcstack[1]}
else
    echo "Only bash and zsh are supported"
    exit 1
fi

NBKITROOT=`dirname ${_SCRIPT_LOCATION}`
NBKITROOT=`readlink -f $NBKITROOT`

# load default python
case "$LOADEDMODULES" in
    *python* )
        # python is loaded
    ;;

    * )
        module load python/3.5-anaconda
    ;;
esac;

# activate python-mpi-bcast
source /usr/common/contrib/bccp/python-mpi-bcast/nersc/activate.sh

# load the specified version
bcast ${NBKITROOT}/nbodykit-dep.tar.gz ${NBKITROOT}/nbodykit-${version}.tar.gz

function srun-nbkit {
    local np=
    if [ "x$1" == "x-n" ]; then
        shift
        np="-n $1"
        shift
    fi
    if [[ "$version" != "v2.0" ]]; then
        srun $np python-mpi /dev/shm/local/bin/nbkit.py $*
    else
        echo "calling 'nbkit.py' is deprecated!"
        return 1
    fi
}