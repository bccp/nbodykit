DIR=`dirname $0`
echo Testing FOF
cd $DIR
[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/nbkit.py FOF -c $fn || exit
done
