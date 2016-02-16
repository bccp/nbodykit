DIR=`dirname $0`
echo 'Testing TidalTensor'
cd $DIR
[ -d ../output ] || mkdir ../output

for fn in *.params; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/nbkit.py TidalTensor @$fn || exit
done
