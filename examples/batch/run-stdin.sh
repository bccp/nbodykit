DIR=`dirname $0`
cd $DIR
[ -d ../output ] || mkdir ../output

echo testing nbkit.py from STDIN ...
echo Some openmpi implementations are buggy causing this test to hang 
echo https://bugzilla.redhat.com/show_bug.cgi?id=1235044
echo use Control-C to stop this one if it hangs.

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
