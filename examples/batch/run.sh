DIR=`dirname $0`
cd $DIR
[ -d ../output ] || mkdir ../output

mpirun -np 6 python ../../bin/nbkit-batch.py FFTPower 2 -c test_power_batch.template -i "los: [x, y, z]" --extras extra.template