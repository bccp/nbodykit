DIR=`dirname $0`
echo Testing RedshiftHistogram
cd $DIR
[ -d ../output ] || mkdir ../output

for fn in *.params; do
    echo testing $fn ...
    python ../../bin/nbkit.py RedshiftHistogram $fn
done