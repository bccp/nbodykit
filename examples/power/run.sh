DIR=`dirname $0`
echo Testing FFTPower
cd $DIR
[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    if [[ $fn == *"corr"* ]]
    then 
        mpirun -n 2 python ../../bin/nbkit.py FFTCorrelation $fn || exit
    elif [[ $fn == *"bianchi"* ]]
    then
        #mpirun -n 2 python ../../bin/nbkit.py BianchiFFTPower $fn || exit
        continue;
    else
        mpirun -n 2 python ../../bin/nbkit.py FFTPower $fn || exit
    fi
done