DIR=`dirname $0`

cd $DIR
[ -d ../output ] || mkdir ../output

for fn in *.params; do
    echo testing $fn ...
    mpirun -n 1 python ../../bin/subhalo.py @$fn || exit
done
