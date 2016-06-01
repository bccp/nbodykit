#! /bin/bash
PREFIX=${NERSC_HOST}

install -d /usr/common/contrib/bccp/nbodykit

install -d /usr/common/contrib/bccp/nbodykit/nersc

install activate.sh /usr/common/contrib/bccp/nbodykit/nersc/

rsync --exclude='*.gz-*' -ar $PREFIX /usr/common/contrib/bccp/nbodykit/nersc/

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
