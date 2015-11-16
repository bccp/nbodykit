source ../../bin/activate.sh
[ -d ../output ] || mkdir ../output
for fn in *.params; do
    echo testing $fn ...
    python ../../bin/trace-halo.py @$fn || exit
done
