#!/bin/bash -l

# install dir
install_dir=/usr/common/contrib/bccp/nbodykit/

# change to a temporary directory
tmpdir=$(mktemp -d)
cd $tmpdir
echo "temporary build directory: " $tmpdir

# activate python-mpi-bcast
source /usr/common/contrib/bccp/python-mpi-bcast/activate.sh

update_tarball()
{
    tarball=$1
    pip_cmd=$2

    # make a build directory
    mkdir build; cd build

    # untar and run the pip update
    if [ -f $install_dir/$tarball ]; then 
        tar -xf $install_dir/$tarball
        pkgdir=$(find . -name 'site-packages')
        pip_output=$(MPICC=cc PYTHONPATH=$pkgdir $pip_cmd)
    else
        pip_output=$(MPICC=cc $pip_cmd)
    fi
    echo "$pip_output"
    
    # remake the tarball?
    if [[ $pip_output == *"Installing collected packages"* ]] || [ ! -f $install_dir/$tarball ]; then
        echo "remaking the tarball '$tarball'..."
        tar -cf $tarball lib
        cp $tarball $install_dir/$tarball
    fi
    cd ..; rm -r build 
}

# build the "latest" source from the HEAD of "master"
tarball=nbodykit-latest.tar.gz
mkdir build; cd build

bundle-pip $tarball git+https://github.com/bccp/nbodykit.git@master
cp $tarball $install_dir/$tarball
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


