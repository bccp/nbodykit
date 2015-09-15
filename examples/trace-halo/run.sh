[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    python ../../trace-halo.py @$fn || exit
done
