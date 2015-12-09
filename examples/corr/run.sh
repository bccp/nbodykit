DIR=`dirname $0`
cd $DIR
[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/corr.py @$fn || exit
done
