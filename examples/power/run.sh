for fn in *.params; do
    python ../../power.py @$fn || exit
done
