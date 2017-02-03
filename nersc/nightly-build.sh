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
    version=$1
    
    if [[ $version != "dev" ]]; then
        
        # increment version
        a=( ${version//./ } ) 
        ((a[1]++))  
        next_version="${a[0]}.${a[1]}"

        # make a build directory
        mkdir build; cd build
    
        # download the right nbodykit source dist
        pip download "nbodykit>=$version<$next_version" --no-deps
    
        # do nothing if no stable version exists
        if [[ ! $(ls -A) ]]; then
            return
        fi
    
        # untar the source tarball and cd
        source_name=$(ls .)
        tar -xvf $source_name
        source_name="${source_name%.tar.gz}"
        a=(${source_name//-/ })
        version=${a[1]}
    
        # get the previous version and remove it
        a=( ${version//./ } ) 
        ((a[2]--))  
        prev_version="${a[0]}.${a[1]}.${a[2]}"
    
        if [ -f $install_dir/nbodykit-$prev_version.tar.gz ]; then
            rm -rf $install_dir/nbodykit-$prev_version.tar.gz
        fi
    fi
    
    # make source and dependency tarballs
    tarballs=("nbodykit-$version.tar.gz" "nbodykit-deps-$version.tar.gz")
    currdir=$(pwd)
    
    i=0
    for tarball in "${tarballs[@]}"; do
        
        if [[ $version == "dev" ]]; then
            if [[ $i == 0 ]]; then
                master="git+https://github.com/bccp/nbodykit.git@master"
                pip_cmd="pip install -U --no-deps --install-option=--prefix=$currdir $master"
            else
                reqs="https://raw.githubusercontent.com/bccp/nbodykit/master/requirements.txt"
                pip_cmd="pip install -U --no-deps --install-option=--prefix=$currdir -r $reqs"
            fi
        elif [[ $i == 0 ]]; then
            pip_cmd="pip install -U --no-deps --install-option=--prefix=$currdir $source_name/"
        else
            pip_cmd="pip install -U --no-deps --install-option=--prefix=$currdir -r $source_name/requirements.txt"
        fi
        
        # untar and run the pip update
        if [ -f $install_dir/$tarball ]; then
            tar -xf $install_dir/$tarball

            # find the right package directory
            case "$LOADEDMODULES" in
              *2.7-anaconda* )
                pkgdir="./lib/python2.7/site-packages"
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
            echo "executing " $pip_cmd
            pip_output=$(MPICC=cc PYTHONPATH=$pkgdir PYTHONUSERBASE=$pkgdir $pip_cmd)
        else
            echo "executing " $pip_cmd
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
        
        ((i++))
    done
    cd ../; rm -r build
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

py_versions=("2.7" "3.5")
nbkit_versions=("0.1" "0.2" "dev")

for py_version in "${py_versions[@]}"; do
    
    load_anaconda $py_version
    
    for nbkit_version in "${nbkit_versions[@]}"; do
    
        update_tarball $nbkit_version
    
    done
done

