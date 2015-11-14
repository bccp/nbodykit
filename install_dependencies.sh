SOURCE=`dirname $0`/depends
LOCAL=`dirname $0`/local
LOCAL=`realpath $LOCAL`

function fail {
    echo Installation of $1 failed
    # list common pitfalls
    # echo Common pitfalls
}

mkdir -p $SOURCE/

function install {
    echo Installing $1 to $LOCAL
    (
    cd $SOURCE
    if ! [ -d $2 ]; then
        git clone $1 $2 
    fi
    cd $2 && git checkout -f master && git pull
    python setup.py install --prefix=$LOCAL
    ) || fail $1
}

install http://github.com/rainwoodman/sharedmem sharedmem
install http://github.com/rainwoodman/MP-sort MP-sort
install http://github.com/rainwoodman/pfft-python pfft-python
install http://github.com/rainwoodman/pypm pypm
install http://github.com/rainwoodman/kdcount kdcount

echo Please add $LOCAL to your python path.
