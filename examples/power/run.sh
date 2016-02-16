DIR=`dirname $0`
cd $DIR
[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    if [[ $fn == *"corr"* ]]
    then 
        mpirun -n 2 python ../../bin/nbkit.py FFTCorrelation -c $fn || exit
    elif [[ $fn == *"bianchi"* ]]
    then
        mpirun -n 2 python ../../bin/nbkit.py BianchiFFTPower -c $fn || exit
    else
        mpirun -n 2 python ../../bin/nbkit.py FFTPower -c $fn || exit
    fi
done

for fn in *.argparse; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/nbkit.py FFTPower @$fn || exit
done
