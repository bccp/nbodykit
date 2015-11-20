DIR=`dirname $0`

cd $DIR
[ -d ../output ] || mkdir ../output

for fn in *.params; do
    echo testing $fn ...
    python ../../bin/subhalo.py @$fn || exit
done
