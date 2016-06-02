#!/bin/bash

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

# get the subcommand and shorten the argument list
subcommand=$1; shift

# get the current git branch
curr_branch=$(git rev-parse --abbrev-ref HEAD)
case "$curr_branch" in 
    master )
       version='stable'
    ;;
    develop )
       version='latest'
    ;;
    * )
       echo "when running build.sh, current git branch should be 'master' or 'develop'"
       exit 1
    ;;
esac

# make the build directory
mkdir -p ${NERSC_HOST}

# activate python-mpi-bcast
source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh

case "$subcommand" in
  all )
    MPICC=cc bundle-pip ${NERSC_HOST}/nbodykit-dep.tar.gz -r ../requirements.txt
    bundle-pip ${NERSC_HOST}/nbodykit-${version}.tar.gz ..
  ;;
  source )
    bundle-pip ${NERSC_HOST}/nbodykit-${version}.tar.gz ..
    ;;
  deps )
    MPICC=cc bundle-pip ${NERSC_HOST}/nbodykit-dep.tar.gz -r ../requirements.txt
    ;;
   * )
    echo "invalid build choice -- choose from 'source' or 'deps'"
    exit 1
esac


