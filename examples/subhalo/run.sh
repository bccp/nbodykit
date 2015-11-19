source ../../bin/activate.sh
for fn in *.params; do
    echo testing $fn ...
    python ../../bin/subhalo.py @$fn || exit
done
