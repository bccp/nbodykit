#!/bin/bash -l

# install dir
install_dir=/usr/common/contrib/bccp/nbodykit/

# change to a temporary directory
tmpdir=$(mktemp -d)
cd $tmpdir

# activate python-mpi-bcast
source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh

update_tarball()
{
    tarball=$1
    pip_cmd=$2

    # make a build directory
    mkdir build; cd build

    # untar and run the pip update
    tar -xf $install_dir/$tarball
    pkgdir=$(find . -name 'site-packages')
    pip_output=$(MPICC=cc PYTHONPATH=$pkgdir $pip_cmd)
    
    # remake the tarball?
    echo "$pip_output"
    if [[ $pip_output == *"Installing collected packages"* ]]; then
        echo "remaking the tarball '$tarball'..."
        tar -cf $tarball lib
        cp $tarball $HOME/test/$tarball
    fi
    cd ..; rm -r build 
}

# build the "latest" source from the HEAD of "master"
$tarball=nbodykit-latest.tar.gz
mkdir build; cd build

bundle-pip $tarball git+https://github.com/bccp/nbodykit.git@master
cp $tarball $HOME/test/$tarball
cd ..; rm -r build

# update the dependencies
tarball=nbodykit-dep.tar.gz
pip_install='pip install -U --no-deps -r https://raw.githubusercontent.com/bccp/nbodykit/master/requirements.txt --prefix .'
update_tarball "${tarball}" "${pip_install}"

# update stable 
tarball=nbodykit-stable.tar.gz
pip_install='pip install -U --no-deps nbodykit --prefix .'
update_tarball "${tarball}" "${pip_install}"

rm -rf $tmpdir


