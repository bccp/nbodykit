source ../../bin/activate.sh

[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    python ../../bin/fof.py @$fn || exit
done
