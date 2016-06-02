#! /bin/bash

if [ "x$1" == "x-h" ] || [ "x$1" == "x" ] ; then
    echo "usage: deploy.sh version"
    exit 1
fi
version=$1
PREFIX=${NERSC_HOST}

# setup build directory
install -d /usr/common/contrib/bccp/nbodykit/builds/$version

# move the activate script to the build dir
install activate.sh /usr/common/contrib/bccp/nbodykit/builds/$version

# copy the necessary tar files
rsync --exclude='*.gz-*' -ar $PREFIX/* /usr/common/contrib/bccp/nbodykit/$version/

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
