DIR=`dirname $0`
cd $DIR
[ -d ../output ] || mkdir ../output

echo testing nbkit-batch.py ...
mpirun -np 6 python ../../bin/nbkit-batch.py FFTPower 2 -c test_power_batch.template -i "los: [x, y, z]" --extras extra.template

echo testing nbkit.py from STDIN ...
mpirun -np 2 python ../../bin/nbkit.py FFTPower <<EOF
mode: 1d
Nmesh: 256
output: ../output/test_power_fastpm_1d.txt

field:
    DataSource: 
       plugin: FastPM
       path: ../data/fastpm_1.0000
    Transfer: [NormalizeDC, RemoveDC, AnisotropicCIC]
EOF
