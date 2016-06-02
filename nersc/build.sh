#!/bin/bash

# parse options
while getopts ":h" opt; do
  case ${opt} in
    h )
      echo "usage:"
      echo "    build.sh -h                      Display this help message."
      echo "    build.sh source <version>        Build only the source."
      echo "    build.sh deps <version>          Build only the dependencies."
      exit 0
      ;;
   \? )
     echo "invalid option: -$OPTARG" 1>&2
     exit 1
     ;;
  esac
done
shift $((OPTIND -1))

subcommand=$1; shift  # Remove build.sh from the argument list
if [ $# != 1 ]
then
    echo "please provide a <version> to build as the 2nd argument"
    exit 1
fi

# activate python-mpi-bcast
source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh

case "$subcommand" in
  
  source )
    version=$1; shift
    mkdir -p ${NERSC_HOST}/${version}
    bundle-pip ${NERSC_HOST}/nbodykit.tar.gz ..
    ;;
  deps )
    version=$1; shift
    mkdir -p ${NERSC_HOST}/${version}
    MPICC=cc bundle-pip ${NERSC_HOST}/${version}/nbodykit-dep.tar.gz -r ../requirements.txt 
    ;;
   * )
    echo "invalid build choice -- choose from 'source' or 'deps'"
    exit 1
esac


