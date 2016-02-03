DIR=`dirname $0`
cd $DIR
[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    if [[ $fn == *"corr"* ]]
    then
        mpirun -n 2 python ../../bin/nbkit.py FFTCorrelation -c $fn || exit
    else
        mpirun -n 2 python ../../bin/nbkit.py PeriodicPower -c $fn || exit
    fi
done

for fn in *.argparse; do
    echo testing $fn ...
    mpirun -n 2 python ../../bin/nbkit.py PeriodicPower @$fn || exit
done
