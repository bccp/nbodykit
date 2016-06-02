#! /bin/bash

if [ "x$1" == "x-h" ] || [ "x$1" == "x" ] ; then
    echo "deploy.sh version"
    exit 1
fi

PREFIX=${NERSC_HOST}

# setup build directory
install -d /usr/common/contrib/bccp/nbodykit/builds/$version

# move the activate script to the build dir
install activate.sh /usr/common/contrib/bccp/nbodykit/builds/$version

# copy the necessary tar files
rsync --exclude='*.gz-*' -ar $PREFIX/* /usr/common/contrib/bccp/nbodykit/builds/$version/*

# setup modulefile directory
install -d /usr/common/contrib/bccp/nbodykit/modulefiles/nbodykit/

# move the modulefile
sed 's/THIS_VERSION/${version}/g' modulefile > /usr/common/contrib/bccp/nbodykit/modulefiles/nbodykit/$version

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
