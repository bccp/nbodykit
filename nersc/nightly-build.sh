#!/bin/bash -l

# install dir
install_dir=/usr/common/contrib/bccp/nbodykit/

# change to a temporary directory
tmpdir=$(mktemp -d)
cd $tmpdir

echo "temporary build directory: " $tmpdir
trap "rm -rf $tmpdir" EXIT

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
        pip_output=$(MPICC=cc PYTHONPATH=$pkgdir PYTHONUSERBASE=$pkgdir $pip_cmd)
    else
        # no tarball so ignore any installed packages with additional -I flag
        pip_output=$(MPICC=cc $pip_cmd -I)
    fi
    cd ..; cd build # avoid stale file handle
    echo "$pip_output"
    
    # remake the tarball?
    if [[ $pip_output == *"Installing collected packages"* ]] || [ ! -f $install_dir/$tarball ]; then
        echo "remaking the tarball '$tarball'..."
        list=
        for dir in bin lib include share; do
            if [ -d $dir ]; then
                list="$list $dir"
            fi
        done
        (
        tar -czf $tarball \
            --exclude='*.html' \
            --exclude='*.jpg' \
            --exclude='*.jpeg' \
            --exclude='*.png' \
            --exclude='*.pyc' \
            --exclude='*.pyo' \
            $list
        ) || exit 1
        (
        install $tarball $install_dir/$tarball 
        ) || exit 1
    fi
    cd ..; rm -r build 
}

# build the "latest" source from the HEAD of "master"
tarball=nbodykit-latest.tar.gz
mkdir build; cd build

bundle-pip $tarball git+https://github.com/bccp/nbodykit.git@master
install $tarball $install_dir/$tarball
cd ..; rm -r build

# update the dependencies
tarball=nbodykit-dep.tar.gz
pip_install="pip install -U --no-deps --install-option=--prefix=$tmpdir/build -r https://raw.githubusercontent.com/bccp/nbodykit/master/requirements.txt"
update_tarball "${tarball}" "${pip_install}" || exit 1

# update stable 
tarball=nbodykit-stable.tar.gz
pip_install="pip install -U --no-deps --install-option=--prefix=$tmpdir/build nbodykit"
update_tarball "${tarball}" "${pip_install}" || exit 1