[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    python ../../power.py @$fn || exit
done
