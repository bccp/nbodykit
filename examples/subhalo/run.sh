for fn in *.params; do
    echo testing $fn ...
    python ../../subhalo.py @$fn || exit
done
