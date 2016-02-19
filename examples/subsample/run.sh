DIR=`dirname $0`
echo Testing Subsample
cd $DIR
[ -d ../output ] || mkdir ../output

for fn in *.params; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/nbkit.py Subsample -c $fn || exit
done
