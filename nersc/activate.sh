#!/bin/bash

function join_by { local IFS="$1"; shift; echo "$*"; }

install_dir=/usr/common/contrib/bccp/nbodykit/
allowed=()
for file in $install_dir/nbodykit-[0-9].[0-9].[0-9].tar.gz; do 
    base=$(basename $file)
    base="${base%.tar.gz}"
    a=(${base//-/ })
    version=${a[1]}
    a=( ${version//./ } ) 
    version="${a[0]}.${a[1]}"
    allowed+=($version)
done

# can supply 0 or 1 argument
if [ "$#" -gt 1 ]; then
    valid=$(join_by "|" ${allowed[@]})
    echo "usage: activate.sh dev|$valid"
    return 1
fi

# if no version provided, use '0.1' and warn
if [ $# -eq 0 ]; then
    version="0.1"
    echo "warning: in the future, please specify the desired nbodykit version as the first argument"
elif [[ "$1" != "dev" ]]; then
    match=0
    for ver in "${allowed[@]}"; do
        if [[ "$1" = "$ver" ]]; then
            match=1
            version=$ver
            break
        fi
    done
    
    if [[ $match = 0 ]]; then
        valid=$(join_by " " ${allowed[@]})
        echo "error: valid version names: dev $valid"
        return 1
    fi
else
    version=$1
fi

# echo the version
echo "loading nbodykit version: " $version

if [[ -n $BASH_VERSION ]]; then
    _SCRIPT_LOCATION=${BASH_SOURCE[0]}
elif [[ -n $ZSH_VERSION ]]; then
    _SCRIPT_LOCATION=${funcstack[1]}
else
    echo "Only bash and zsh are supported"
    return 1
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

# bcast the tarballs
for tarball in ${NBKITROOT}/nbodykit-*${version}*.tar.gz; do
    echo "bcasting " $tarball
    bcast $tarball
done


function srun-nbkit {
    local np=
    if [ "x$1" == "x-n" ]; then
        shift
        np="-n $1"
        shift
    fi
    if [[ "$version" == "0.1" ]]; then
        srun $np python-mpi /dev/shm/local/bin/nbkit.py $*
    else
        echo "error: calling 'nbkit.py' is deprecated!"
        echo "usage: srun -n NTASKS python-mpi job_script.py"
        echo "see nbodykit/tests for examples using new 'nbodykit.lab' syntax"
        return 1
        
    fi
}