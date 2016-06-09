#! /bin/bash

if [ "x$1" == "x-h" ] || [ "x$1" == "x" ] ; then
    echo "usage: deploy.sh <latest|stable>"
    exit 1
fi


# check input version
if [[ "$1" -ne "stable" || "$1" -ne "latest" ]]; then
    echo "valid version names are 'stable' and 'latest'"
    exit 1
fi

version=$1
PREFIX=${NERSC_HOST}

# move the activate script to the build dir
install activate.sh /usr/common/contrib/bccp/nbodykit/

# copy the necessary tar files
rsync --exclude='*.gz-*' -ar $PREFIX/nbodykit-dep.tar.gz /usr/common/contrib/bccp/nbodykit/
rsync --exclude='*.gz-*' -ar $PREFIX/nbodykit-$version.tar.gz /usr/common/contrib/bccp/nbodykit/

function tree {
    SEDMAGIC='s;[^/]*/;|____;g;s;____|; |;g'

    if [ "$#" -gt 0 ] ; then
       dirlist="$@"
    else
       dirlist="."
    fi

    for x in $dirlist; do
         find "$x" -printf "%p@%t\n" | sed -e "$SEDMAGIC"|awk -F @ '{printf("%-40s %s\n", $1, $2)}'
    done

}

echo "Done. Tree of files... "
(
cd /usr/common/contrib/bccp/nbodykit/;
tree .
)
