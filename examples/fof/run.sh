[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    python ../../fof.py @$fn || exit
done
