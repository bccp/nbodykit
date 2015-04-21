git submodule init
git submodule update
for dir in MP-sort kdcount pfft-python pypm sharedmem; do
    (
        cd extern/$dir;
        python setup.py build_ext --inplace
    ) || exit
done

