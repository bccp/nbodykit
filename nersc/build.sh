#!/bin/bash

# shell script to either build the dependences or the source
# if source is being built, the local git repo will be tarred
# outputs: $NERSC_HOST/nbodykit-dep.tar.gz or $NERSC_HOST/nbodykit.tar.gz

# print the usage
while getopts ":h" opt; do
  case ${opt} in
    h )
      echo "usage:"
      echo "    build.sh -h            Display this help message."
      echo "    build.sh all           Build both the source and the dependencies."
      echo "    build.sh source        Build only the source."
      echo "    build.sh deps          Build only the dependencies."
      exit 0
      ;;
   \? )
     echo "invalid option: -$OPTARG" 1>&2
     exit 1
     ;;
  esac
done
shift $((OPTIND -1))

# make sure we are running from "nersc" directory
if [ ! -f ../requirements.txt ] || [ $(basename "$PWD") != "nersc" ]; then
    echo "please call this script from the 'nbodykit/nersc' directory"
    exit 1
fi


# get the subcommand and shorten the argument list
subcommand=$1; shift

# make the build directory
mkdir -p ${NERSC_HOST}

# activate python-mpi-bcast
source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh

case "$subcommand" in
  all )
    MPICC=cc bundle-pip ${NERSC_HOST}/nbodykit-dep.tar.gz -r ../requirements.txt
    bundle-pip ${NERSC_HOST}/nbodykit.tar.gz ..
  ;;
  source )
    bundle-pip ${NERSC_HOST}/nbodykit.tar.gz ..
    ;;
  deps )
    MPICC=cc bundle-pip ${NERSC_HOST}/nbodykit-dep.tar.gz -r ../requirements.txt
    ;;
   * )
    echo "invalid build choice -- choose from 'source', 'deps', 'all'"
    exit 1
esac


