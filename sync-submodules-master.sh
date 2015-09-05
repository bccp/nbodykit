for dir in MP-sort kdcount pfft-python pypm sharedmem; do
    (
        echo $dir
        cd extern/$dir;
        git reset --hard
        git checkout master
        git pull
    ) || exit
done

