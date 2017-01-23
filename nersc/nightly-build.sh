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
        
        # find the right package directory
        case "$LOADEDMODULES" in
          *2.7-anaconda* )
            pkgdir="./lib/python2.7/site-packages"
            ;;
          *3.4-anaconda* )
            pkgdir="./lib/python3.4/site-packages"
            ;;
          *3.5-anaconda* )
            pkgdir="./lib/python3.5/site-packages"
            ;;
          * )
            echo "cannot find correct package directorys"
            exit 1
          ;;
        esac;
        
        # run the pip command        
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

load_anaconda() 
{
    version=$1
    module unload python
    echo "loading python/$version-anaconda..."
    case "$version" in
      "2.7" )
       module load python/2.7-anaconda
       source activate /usr/common/contrib/bccp/nbodykit/build-envs/2.7-nbodykit-base
        ;;
      "3.5" )
        module load python/3.5-anaconda
        source activate /usr/common/contrib/bccp/nbodykit/build-envs/3.5-nbodykit-base
        ;;
      * )
        echo "supported python anaconda modules are 2.7, 3.5"
        exit 1
      ;;
    esac;
}

versions=("2.7" "3.5")

for version in "${versions[@]}"; do
    
    # load the right anaconda version
    load_anaconda $version
    
    # build the "latest" source from the HEAD of "master"
    tarball=nbodykit-latest.tar.gz
    master="git+https://github.com/bccp/nbodykit.git@master"
    pip_install="pip install -U --no-deps --install-option=--prefix=$tmpdir/build $master"
    update_tarball "${tarball}" "${pip_install}" || exit 1
    
    # build v2.0 source from the HEAD of "v2.0"
    tarball=nbodykit-v2.0.tar.gz
    master="git+https://github.com/bccp/nbodykit.git@v2.0"
    pip_install="pip install -U --no-deps --install-option=--prefix=$tmpdir/build $master"
    update_tarball "${tarball}" "${pip_install}" || exit 1

    # update the dependencies
    tarball=nbodykit-dep.tar.gz
    reqs="https://raw.githubusercontent.com/bccp/nbodykit/v2.0/requirements.txt"
    pip_install="pip install -U --no-deps --install-option=--prefix=$tmpdir/build -r $reqs"
    update_tarball "${tarball}" "${pip_install}" || exit 1
    
    # update stable
    tarball=nbodykit-stable.tar.gz
    pip_install="pip install -U --no-deps --install-option=--prefix=$tmpdir/build nbodykit"
    update_tarball "${tarball}" "${pip_install}" || exit 1
done
