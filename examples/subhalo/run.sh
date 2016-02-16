DIR=`dirname $0`
echo Testinb FOF6D
cd $DIR
[ -d ../output ] || mkdir ../output

for fn in *.params; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/nbkit.py FOF6D @$fn || exit
done
