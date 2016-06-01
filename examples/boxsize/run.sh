DIR=`dirname $0`
echo Testing TestBoxSize
cd $DIR
[ -d ../output ] || mkdir ../output

for fn in *.params; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/nbkit.py TestBoxSize $fn
done